# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import heapq
import os
from typing import TYPE_CHECKING

import numpy as np
import omni.log
import torch
import trimesh
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.terrains.utils import color_meshes_by_height, find_flat_patches
from isaaclab.utils.dict import dict_to_md5_hash
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.timer import Timer
from isaaclab.utils.warp import convert_to_warp_mesh

from go2_lidar.terrain.utils import GridMapNavigation, MazeGridNavigation, _create_composite_obstacle_nav

if TYPE_CHECKING:
    from isaaclab.terrains.sub_terrain_cfg import FlatPatchSamplingCfg, SubTerrainBaseCfg
    from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


class NavTerrainGenerator:
    r"""Terrain generator to handle different terrain generation functions.
    Specifically designed for training the navigation policy.

    The terrains are represented as meshes. These are obtained either from height fields or by using the
    `trimesh <https://trimsh.org/trimesh.html>`__ library. The height field representation is more
    flexible, but it is less computationally and memory efficient than the trimesh representation.

    All terrain generation functions take in the argument :obj:`difficulty` which determines the complexity
    of the terrain. The difficulty is a number between 0 and 1, where 0 is the easiest and 1 is the hardest.
    In most cases, the difficulty is used for linear interpolation between different terrain parameters.
    For example, in a pyramid stairs terrain the step height is interpolated between the specified minimum
    and maximum step height.

    Each sub-terrain has a corresponding configuration class that can be used to specify the parameters
    of the terrain. The configuration classes are inherited from the :class:`SubTerrainBaseCfg` class
    which contains the common parameters for all terrains.

    If a curriculum is used, the terrains are generated based on their difficulty parameter.
    The difficulty is varied linearly over the number of rows (i.e. along x) with a small random value
    added to the difficulty to ensure that the columns with the same sub-terrain type are not exactly
    the same. The difficulty parameter for a sub-terrain at a given row is calculated as:

    .. math::

        \text{difficulty} = \frac{\text{row_id} + \eta}{\text{num_rows}} \times (\text{upper} - \text{lower}) + \text{lower}

    where :math:`\eta\sim\mathcal{U}(0, 1)` is a random perturbation to the difficulty, and
    :math:`(\text{lower}, \text{upper})` is the range of the difficulty parameter, specified using the
    :attr:`~TerrainGeneratorCfg.difficulty_range` parameter.

    If a curriculum is not used, the terrains are generated randomly. In this case, the difficulty parameter
    is randomly sampled from the specified range, given by the :attr:`~TerrainGeneratorCfg.difficulty_range` parameter:

    .. math::

        \text{difficulty} \sim \mathcal{U}(\text{lower}, \text{upper})

    If the :attr:`~TerrainGeneratorCfg.flat_patch_sampling` is specified for a sub-terrain, flat patches are sampled
    on the terrain. These can be used for spawning robots, targets, etc. The sampled patches are stored
    in the :obj:`flat_patches` dictionary. The key specifies the intention of the flat patches and the
    value is a tensor containing the flat patches for each sub-terrain.

    If the flag :attr:`~TerrainGeneratorCfg.use_cache` is set to True, the terrains are cached based on their
    sub-terrain configurations. This means that if the same sub-terrain configuration is used
    multiple times, the terrain is only generated once and then reused. This is useful when
    generating complex sub-terrains that take a long time to generate.

    .. attention::

        The terrain generation has its own seed parameter. This is set using the :attr:`TerrainGeneratorCfg.seed`
        parameter. If the seed is not set and the caching is disabled, the terrain generation may not be
        completely reproducible.

    """

    terrain_mesh: trimesh.Trimesh
    """A single trimesh.Trimesh object for all the generated sub-terrains."""
    terrain_meshes: list[trimesh.Trimesh]
    """List of trimesh.Trimesh objects for all the generated sub-terrains."""
    terrain_origins: np.ndarray
    """The origin of each sub-terrain. Shape is (num_rows, num_cols, 3)."""
    flat_patches: dict[str, torch.Tensor]
    """A dictionary of sampled valid (flat) patches for each sub-terrain.

    The dictionary keys are the names of the flat patch sampling configurations. This maps to a
    tensor containing the flat patches for each sub-terrain. The shape of the tensor is
    (num_rows, num_cols, num_patches, 3).

    For instance, the key "root_spawn" maps to a tensor containing the flat patches for spawning an asset.
    Similarly, the key "target_spawn" maps to a tensor containing the flat patches for setting targets.
    """

    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu"):
        """Initialize the terrain generator.

        Args:
            cfg: Configuration for the terrain generator.
            device: The device to use for the flat patches tensor.
        """
        # check inputs
        if len(cfg.sub_terrains) == 0:
            raise ValueError("No sub-terrains specified! Please add at least one sub-terrain.")
        # store inputs
        self.cfg = cfg
        self.device = device

        # set common values to all sub-terrains config
        from isaaclab.terrains.height_field import HfTerrainBaseCfg  # prevent circular import

        for sub_cfg in self.cfg.sub_terrains.values():
            # size of all terrains
            sub_cfg.size = self.cfg.size
            # params for height field terrains
            if isinstance(sub_cfg, HfTerrainBaseCfg):
                sub_cfg.horizontal_scale = self.cfg.horizontal_scale
                sub_cfg.vertical_scale = self.cfg.vertical_scale
                sub_cfg.slope_threshold = self.cfg.slope_threshold

        # throw a warning if the cache is enabled but the seed is not set
        if self.cfg.use_cache and self.cfg.seed is None:
            omni.log.warn(
                "Cache is enabled but the seed is not set. The terrain generation will not be reproducible."
                " Please set the seed in the terrain generator configuration to make the generation reproducible."
            )

        # if the seed is not set, we assume there is a global seed set and use that.
        # this ensures that the terrain is reproducible if the seed is set at the beginning of the program.
        if self.cfg.seed is not None:
            seed = self.cfg.seed
        else:
            seed = np.random.get_state()[1][0]
        # set the seed for reproducibility
        # note: we create a new random number generator to avoid affecting the global state
        #  in the other places where random numbers are used.
        self.np_rng = np.random.default_rng(seed)

        # buffer for storing valid patches
        self.flat_patches = {}
        # create a list of all sub-terrains
        self.terrain_meshes = list()
        self.terrain_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3))

        # parse configuration and add sub-terrains
        # create terrains based on curriculum or randomly
        if self.cfg.curriculum:
            with Timer("[INFO] Generating terrains based on curriculum took"):
                self._generate_curriculum_terrains()
        else:
            with Timer("[INFO] Generating terrains randomly took"):
                self._generate_random_terrains()
        # add a border around the terrains
        self._add_terrain_border()
        # add curriculum obstacles
        self._add_curriculum_obstacles()
        # combine all the sub-terrains into a single mesh
        self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)

        # color the terrain mesh
        if self.cfg.color_scheme == "height":
            self.terrain_mesh = color_meshes_by_height(self.terrain_mesh)
        elif self.cfg.color_scheme == "random":
            self.terrain_mesh.visual.vertex_colors = self.np_rng.choice(
                range(256), size=(len(self.terrain_mesh.vertices), 4)
            )
        elif self.cfg.color_scheme == "none":
            pass
        else:
            raise ValueError(f"Invalid color scheme: {self.cfg.color_scheme}.")

        # offset the entire terrain and origins so that it is centered
        # -- terrain mesh
        transform = np.eye(4)
        transform[:2, -1] = -self.cfg.size[0] * self.cfg.num_rows * 0.5, -self.cfg.size[1] * self.cfg.num_cols * 0.5
        self.terrain_mesh.apply_transform(transform)
        # -- terrain origins
        self.terrain_origins += transform[:3, -1]
        # -- valid patches
        for name, value in self.flat_patches.items():
            if not isinstance(value, torch.Tensor):
                continue
            self.flat_patches[name][..., 0] += -self.cfg.size[0] * self.cfg.num_rows * 0.5
            self.flat_patches[name][..., 1] += -self.cfg.size[1] * self.cfg.num_cols * 0.5

    def __str__(self):
        """Return a string representation of the terrain generator."""
        msg = "Terrain Generator:"
        msg += f"\n\tSeed: {self.cfg.seed}"
        msg += f"\n\tNumber of rows: {self.cfg.num_rows}"
        msg += f"\n\tNumber of columns: {self.cfg.num_cols}"
        msg += f"\n\tSub-terrain size: {self.cfg.size}"
        msg += f"\n\tSub-terrain types: {list(self.cfg.sub_terrains.keys())}"
        msg += f"\n\tCurriculum: {self.cfg.curriculum}"
        msg += f"\n\tDifficulty range: {self.cfg.difficulty_range}"
        msg += f"\n\tColor scheme: {self.cfg.color_scheme}"
        msg += f"\n\tUse cache: {self.cfg.use_cache}"
        if self.cfg.use_cache:
            msg += f"\n\tCache directory: {self.cfg.cache_dir}"

        return msg

    """
    Terrain generator functions.
    """

    def _generate_random_terrains(self):
        """Add terrains based on randomly sampled difficulty parameter."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        # randomly sample sub-terrains
        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            # coordinate index of the sub-terrain
            (sub_row, sub_col) = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            # randomly sample terrain index
            sub_index = self.np_rng.choice(len(proportions), p=proportions)
            # randomly sample difficulty parameter
            difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
            # generate terrain
            mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_index])
            # add to sub-terrains
            self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index])

    def _generate_curriculum_terrains(self):
        """Add terrains based on the difficulty parameter."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        # find the sub-terrain index for each column
        # we generate the terrains based on their proportion (not randomly sampled)
        sub_indices = []
        for index in range(self.cfg.num_cols):
            sub_index = np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
            sub_indices.append(sub_index)
        sub_indices = np.array(sub_indices, dtype=np.int32)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        # curriculum-based sub-terrains
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                # vary the difficulty parameter linearly over the number of rows
                # note: based on the proportion, multiple columns can have the same sub-terrain type.
                #  Thus to increase the diversity along the rows, we add a small random value to the difficulty.
                #  This ensures that the terrains are not exactly the same. For example, if the
                #  the row index is 2 and the number of rows is 10, the nominal difficulty is 0.2.
                #  We add a small random value to the difficulty to make it between 0.2 and 0.3.
                lower, upper = self.cfg.difficulty_range
                difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                difficulty = lower + (upper - lower) * difficulty
                # generate terrain
                mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[sub_indices[sub_col]])
                # add to sub-terrains
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_indices[sub_col]])

    """
    Internal helper functions.
    """

    def _add_terrain_border(self):
        """Add a surrounding border over all the sub-terrains into the terrain meshes."""
        # border parameters
        border_size = (
            self.cfg.num_rows * self.cfg.size[0] + 2 * self.cfg.border_width,
            self.cfg.num_cols * self.cfg.size[1] + 2 * self.cfg.border_width,
        )
        inner_size = (self.cfg.num_rows * self.cfg.size[0], self.cfg.num_cols * self.cfg.size[1])
        border_center = (
            self.cfg.num_rows * self.cfg.size[0] / 2,
            self.cfg.num_cols * self.cfg.size[1] / 2,
            -self.cfg.border_height / 2,
        )
        # border mesh
        border_meshes = make_border(border_size, inner_size, height=abs(self.cfg.border_height), position=border_center)
        border = trimesh.util.concatenate(border_meshes)
        # update the faces to have minimal triangles
        selector = ~(np.asarray(border.triangles)[:, :, 2] < -0.1).any(1)
        border.update_faces(selector)
        # add the border to the list of meshes
        self.terrain_meshes.append(border)

    def _add_sub_terrain(
        self, mesh: trimesh.Trimesh, origin: np.ndarray, row: int, col: int, sub_terrain_cfg: SubTerrainBaseCfg
    ):
        """Add input sub-terrain to the list of sub-terrains.

        This function adds the input sub-terrain mesh to the list of sub-terrains and updates the origin
        of the sub-terrain in the list of origins. It also samples flat patches if specified.

        Args:
            mesh: The mesh of the sub-terrain.
            origin: The origin of the sub-terrain.
            row: The row index of the sub-terrain.
            col: The column index of the sub-terrain.
        """
        # sample flat patches if specified
        if sub_terrain_cfg.flat_patch_sampling is not None:
            omni.log.info(f"Sampling flat patches for sub-terrain at (row, col):  ({row}, {col})")
            # convert the mesh to warp mesh
            wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
            # sample flat patches based on each patch configuration for that sub-terrain
            for name, patch_cfg in sub_terrain_cfg.flat_patch_sampling.items():
                patch_cfg: FlatPatchSamplingCfg
                # create the flat patches tensor (if not already created)
                if name not in self.flat_patches:
                    self.flat_patches[name] = torch.zeros(
                        (self.cfg.num_rows, self.cfg.num_cols, patch_cfg.num_patches, 3), device=self.device
                    )
                # add the flat patches to the tensor
                self.flat_patches[name][row, col] = find_flat_patches(
                    wp_mesh=wp_mesh,
                    origin=origin,
                    num_patches=patch_cfg.num_patches,
                    patch_radius=patch_cfg.patch_radius,
                    x_range=patch_cfg.x_range,
                    y_range=patch_cfg.y_range,
                    z_range=patch_cfg.z_range,
                    max_height_diff=patch_cfg.max_height_diff,
                )

        # transform the mesh to the correct position
        transform = np.eye(4)
        transform[0:2, -1] = (row + 0.5) * self.cfg.size[0], (col + 0.5) * self.cfg.size[1]
        mesh.apply_transform(transform)
        # add mesh to the list
        self.terrain_meshes.append(mesh)
        # add origin to the list
        self.terrain_origins[row, col] = origin + transform[:3, -1]

    def _get_terrain_mesh(self, difficulty: float, cfg: SubTerrainBaseCfg) -> tuple[trimesh.Trimesh, np.ndarray]:
        """Generate a sub-terrain mesh based on the input difficulty parameter.

        If caching is enabled, the sub-terrain is cached and loaded from the cache if it exists.
        The cache is stored in the cache directory specified in the configuration.

        .. Note:
            This function centers the 2D center of the mesh and its specified origin such that the
            2D center becomes :math:`(0, 0)` instead of :math:`(size[0] / 2, size[1] / 2).

        Args:
            difficulty: The difficulty parameter.
            cfg: The configuration of the sub-terrain.

        Returns:
            The sub-terrain mesh and origin.
        """
        # copy the configuration
        cfg = cfg.copy()
        # add other parameters to the sub-terrain configuration
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        # generate hash for the sub-terrain
        sub_terrain_hash = dict_to_md5_hash(cfg.to_dict())
        # generate the file name
        sub_terrain_cache_dir = os.path.join(self.cfg.cache_dir, sub_terrain_hash)
        sub_terrain_obj_filename = os.path.join(sub_terrain_cache_dir, "mesh.obj")
        sub_terrain_csv_filename = os.path.join(sub_terrain_cache_dir, "origin.csv")
        sub_terrain_meta_filename = os.path.join(sub_terrain_cache_dir, "cfg.yaml")

        # check if hash exists - if true, load the mesh and origin and return
        if self.cfg.use_cache and os.path.exists(sub_terrain_obj_filename):
            # load existing mesh
            mesh = trimesh.load_mesh(sub_terrain_obj_filename, process=False)
            origin = np.loadtxt(sub_terrain_csv_filename, delimiter=",")
            # return the generated mesh
            return mesh, origin

        # generate the terrain
        meshes, origin = cfg.function(difficulty, cfg)
        mesh = trimesh.util.concatenate(meshes)
        # offset mesh such that they are in their center
        transform = np.eye(4)
        transform[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        # change origin to be in the center of the sub-terrain
        origin += transform[0:3, -1]

        # if caching is enabled, save the mesh and origin
        if self.cfg.use_cache:
            # create the cache directory
            os.makedirs(sub_terrain_cache_dir, exist_ok=True)
            # save the data
            mesh.export(sub_terrain_obj_filename)
            np.savetxt(sub_terrain_csv_filename, origin, delimiter=",", header="x,y,z")
            dump_yaml(sub_terrain_meta_filename, cfg)
        # return the generated mesh
        return mesh, origin

    def _add_curriculum_obstacles(self):
        """Add curriculum obstacles for the navigation policy."""
        if self.cfg.num_rows != 10 or self.cfg.num_cols != 30:
            raise ValueError(
                "Curriculum obstacles are only supported for 10 rows and 30 columns. "
                f"Current configuration has {self.cfg.num_rows} rows and {self.cfg.num_cols} columns."
            )
        self._num_curriculum = self.cfg.num_rows
        self._col_per_type = self.cfg.num_cols // 3

        self._num_flat_patches = 1000
        self.flat_patches["general"] = torch.zeros(self._num_flat_patches, 3, device=self.device)
        self.flat_patches["dead_end"] = torch.zeros(self._num_flat_patches, 3, device=self.device)
        self.flat_patches["detour"] = torch.zeros(self._num_flat_patches, 3, device=self.device)

        #
        # curriculum: general movement
        #
        terrain_size = (self.cfg.size[0] * self._num_curriculum, self.cfg.size[1] * self._col_per_type)
        patches, patch_goals, guide_generator = self._create_general_nav(
            terrain_size[0] * 0.5, terrain_size[1] * 0.5, 0.0, 1.0, self.terrain_meshes
        )
        self.flat_patches["general_start"] = torch.cat(
            [
                torch.tensor(patches, dtype=torch.float32, device=self.device),
                torch.zeros(patches.shape[0], 1, device=self.device),
            ],
            dim=-1,
        )
        self.flat_patches["general_goal"] = torch.cat(
            [
                torch.tensor(patch_goals, dtype=torch.float32, device=self.device),
                torch.zeros(patch_goals.shape[0], 1, device=self.device),
            ],
            dim=-1,
        )
        self.flat_patches["general_guide_generator"] = guide_generator

        #
        # curriculum: getting out of dead end
        #

        terrain_size = (self.cfg.size[0] * self._num_curriculum, self.cfg.size[1] * self._col_per_type)
        grid_row = 1
        grid_col = 1
        obst_count = grid_row * grid_col
        cell_size_row = terrain_size[0] / grid_row
        cell_size_col = terrain_size[1] / grid_col
        obst_centers = np.zeros((obst_count, 2))
        obst_centers[:, 0] = np.arange(obst_count) // grid_col * cell_size_row + cell_size_row / 2.0
        obst_centers[:, 1] = np.arange(obst_count) % grid_col * cell_size_col + cell_size_col / 2.0
        obst_centers[:, 0] += np.random.uniform(-0.5, 0.5, obst_count)
        obst_centers[:, 1] += np.random.uniform(-0.5, 0.5, obst_count)
        obst_centers[:, 1] += terrain_size[1]

        # add random composite obstacles
        patches, patch_goals, guide_generator = self._create_dead_end_nav(
            obst_centers[0, 0], obst_centers[0, 1], 0.0, 1.0, self.terrain_meshes
        )
        self.flat_patches["dead_end_start"] = torch.cat(
            [
                torch.tensor(patches, dtype=torch.float32, device=self.device),
                torch.zeros(patches.shape[0], 1, device=self.device),
            ],
            dim=-1,
        )
        self.flat_patches["dead_end_goal"] = torch.cat(
            [
                torch.tensor(patch_goals, dtype=torch.float32, device=self.device),
                torch.zeros(*patch_goals.shape[:-1], 1, device=self.device),
            ],
            dim=-1,
        )
        self.flat_patches["dead_end_guide_generator"] = guide_generator

        #
        # curriculum: long detour
        #

        terrain_size = (self.cfg.size[0] * self._num_curriculum, self.cfg.size[1] * self._col_per_type)
        grid_row = 1
        grid_col = 1
        obst_count = grid_row * grid_col
        cell_size_row = terrain_size[0] / grid_row
        cell_size_col = terrain_size[1] / grid_col
        obst_centers = np.zeros((obst_count, 2))
        obst_centers[:, 0] = np.arange(obst_count) // grid_col * cell_size_row + cell_size_row / 2.0
        obst_centers[:, 1] = np.arange(obst_count) % grid_col * cell_size_col + cell_size_col / 2.0
        obst_centers[:, 0] += np.random.uniform(-0.5, 0.5, obst_count)
        obst_centers[:, 1] += np.random.uniform(-0.5, 0.5, obst_count)
        obst_centers[:, 1] += terrain_size[1] * 2

        # add random composite obstacles
        patches, patch_goals, guide_generator = self._create_detour_corridor_nav(
            obst_centers[0, 0], obst_centers[0, 1], 0.0, 1.0, self.terrain_meshes
        )
        self.flat_patches["detour_start"] = torch.cat(
            [
                torch.tensor(patches, dtype=torch.float32, device=self.device),
                torch.zeros(patches.shape[0], 1, device=self.device),
            ],
            dim=-1,
        )
        self.flat_patches["detour_goal"] = torch.cat(
            [
                torch.tensor(patch_goals, dtype=torch.float32, device=self.device),
                torch.zeros(*patch_goals.shape[:-1], 1, device=self.device),
            ],
            dim=-1,
        )
        self.flat_patches["detour_guide_generator"] = guide_generator

    def _generate_maze_grid(self, width, height):
        """Generate a maze using recursive backtracking algorithm.

        Returns a 2D array where each cell contains a 4-bit value:
        bit 3 (0b1000): top wall
        bit 2 (0b0100): right wall
        bit 1 (0b0010): bottom wall
        bit 0 (0b0001): left wall
        """
        # Initialize maze with all walls
        maze = np.full((width, height), 0b1111, dtype=np.uint8)

        # Track visited cells
        visited = np.zeros((width, height), dtype=bool)

        # Stack for backtracking
        stack = []

        # Start from random cell
        current = (self.np_rng.integers(width), self.np_rng.integers(height))
        visited[current] = True

        # Directions: top, right, bottom, left
        directions = [(0, 1, 0b1000, 0b0010), (1, 0, 0b0100, 0b0001), (0, -1, 0b0010, 0b1000), (-1, 0, 0b0001, 0b0100)]

        while True:
            x, y = current
            neighbors = []

            # Find unvisited neighbors
            for dx, dy, wall_current, wall_neighbor in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and not visited[nx, ny]:
                    neighbors.append((nx, ny, wall_current, wall_neighbor))

            if neighbors:
                # Choose random unvisited neighbor
                nx, ny, wall_current, wall_neighbor = neighbors[self.np_rng.integers(len(neighbors))]

                # Remove walls between current and neighbor
                maze[x, y] &= ~wall_current
                maze[nx, ny] &= ~wall_neighbor

                # Mark neighbor as visited
                visited[nx, ny] = True

                # Push current to stack and move to neighbor
                stack.append(current)
                current = (nx, ny)
            elif stack:
                # Backtrack
                current = stack.pop()
            else:
                # Done
                break

        return maze

    def _create_detour_corridor_nav(self, center_x, center_y, base_rotation, difficulty, meshes_list):
        """Creates a randomly generated maze using recursive backtracking.

        Args:
            center_x: X coordinate of the grid center
            center_y: Y coordinate of the grid center
            base_rotation: Base rotation to apply to the entire structure
            difficulty: Difficulty level (0-1), affects obstacle density
            meshes_list: List to append obstacle meshes to

        Returns:
            spawn_positions_world: numpy array of shape (N, 2) containing spawn positions
            goal_positions_world: numpy array of shape (N, 2) containing goal positions
            navigation: MazeGridNavigation object for path planning
        """
        temp_meshes = []

        def create_wall_pillars(start_pos, end_pos):
            """Creates a wall from start_pos to end_pos using small pillars."""
            wall_length = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
            if wall_length < 0.01:
                return []

            wall_direction = (np.array(end_pos) - np.array(start_pos)) / wall_length

            pillar_size = self.np_rng.uniform(0.1, 0.2)
            pillar_spacing = pillar_size * 1.2
            num_pillars = int(wall_length / pillar_spacing) + 1

            wall_pillars = []

            for i in range(num_pillars):
                pillar_offset = 0
                if self.np_rng.random() < 0.2:
                    pillar_offset = self.np_rng.uniform(0.1, 0.2)

                pillar_height = self.np_rng.uniform(0.5, 1.5)
                pillar_pos = np.array(start_pos) + wall_direction * (i * pillar_spacing)

                current_pillar_size = pillar_size * self.np_rng.uniform(0.9, 1.1)
                extents = (current_pillar_size, current_pillar_size, pillar_height)

                box = trimesh.creation.box(extents=extents)
                box.apply_translation(
                    [
                        pillar_pos[0] + wall_direction[0] * pillar_offset,
                        pillar_pos[1] + wall_direction[1] * pillar_offset,
                        pillar_height / 2.0,
                    ]
                )

                wall_pillars.append(box)

            return wall_pillars

        # Maze parameters
        cell_size = 1.75
        maze_width = 40
        maze_height = 40

        # Generate maze structure
        maze = self._generate_maze_grid(maze_width, maze_height)

        # Create boundary walls
        terrain_size = (maze_width * cell_size, maze_height * cell_size)
        meshes = create_wall_pillars((0.0, 0.0), (0.0, terrain_size[0]))
        temp_meshes.extend(meshes)
        meshes = create_wall_pillars((0.0, 0.0), (terrain_size[1], 0.0))
        temp_meshes.extend(meshes)
        meshes = create_wall_pillars((0.0, terrain_size[0]), (terrain_size[1], terrain_size[0]))
        temp_meshes.extend(meshes)
        meshes = create_wall_pillars((terrain_size[1], 0.0), (terrain_size[1], terrain_size[0]))
        temp_meshes.extend(meshes)

        # Create walls based on maze
        # Match test_terrain_generator.py convention: i=X, j=Y
        for i in range(maze_width):
            for j in range(maze_height):
                walls = maze[i, j]

                # Cell (i, j) center position
                cell_center_x = (i + 0.5) * cell_size
                cell_center_y = (j + 0.5) * cell_size

                # Top wall - horizontal wall at top of cell
                if walls & 0b1000:
                    wall_start = (cell_center_x - cell_size / 2, cell_center_y + cell_size / 2)
                    wall_end = (cell_center_x + cell_size / 2, cell_center_y + cell_size / 2)
                    temp_meshes.extend(create_wall_pillars(wall_start, wall_end))

                # Right wall - vertical wall at right of cell
                if walls & 0b0100:
                    wall_start = (cell_center_x + cell_size / 2, cell_center_y - cell_size / 2)
                    wall_end = (cell_center_x + cell_size / 2, cell_center_y + cell_size / 2)
                    temp_meshes.extend(create_wall_pillars(wall_start, wall_end))

                # Bottom wall - horizontal wall at bottom of cell
                if walls & 0b0010:
                    wall_start = (cell_center_x - cell_size / 2, cell_center_y - cell_size / 2)
                    wall_end = (cell_center_x + cell_size / 2, cell_center_y - cell_size / 2)
                    temp_meshes.extend(create_wall_pillars(wall_start, wall_end))

                # Left wall - vertical wall at left of cell
                if walls & 0b0001:
                    wall_start = (cell_center_x - cell_size / 2, cell_center_y - cell_size / 2)
                    wall_end = (cell_center_x - cell_size / 2, cell_center_y + cell_size / 2)
                    temp_meshes.extend(create_wall_pillars(wall_start, wall_end))

        # Center offset
        offset_x = -terrain_size[0] / 2
        offset_y = -terrain_size[1] / 2

        # Find all free cells (all cells in maze are free)
        free_cells = []
        for i in range(maze_width):
            for j in range(maze_height):
                world_x = (i + 0.5) * cell_size + offset_x
                world_y = (j + 0.5) * cell_size + offset_y
                free_cells.append((i, j, world_x, world_y))

        free_cells = np.array(free_cells)

        # Randomly sample 200 goal positions
        num_pairs = 1000
        goal_indices = self.np_rng.choice(len(free_cells), num_pairs, replace=False)
        goals_local = []
        goal_positions_grid = []

        for idx in goal_indices:
            mx, my, wx, wy = free_cells[idx]
            goals_local.append([wx, wy])
            goal_positions_grid.append((int(mx), int(my)))

        goals_local = np.array(goals_local)

        # Create single navigation object (will be used for spawn selection AND returned)
        navigation = MazeGridNavigation(
            maze_width,
            maze_height,
            goal_positions_grid,
            cell_size,
            offset_x,
            offset_y,
            center_x,
            center_y,
            base_rotation,
            device=self.device,
        )
        navigation.add_walls(maze)

        # For each goal, find valid spawn positions
        min_straight_dist = 5.0
        max_straight_dist = 7.0
        min_path_dist = 5.0
        max_path_dist = 15.0
        spawn_positions_local = []

        for goal_idx, goal_local in enumerate(goals_local):
            goal_mx, goal_my = goal_positions_grid[goal_idx]

            # Find candidate spawn positions
            candidates = []
            for spawn_cell in free_cells:
                spawn_mx, spawn_my, spawn_wx, spawn_wy = spawn_cell

                # Check straight-line distance (must be between min and max)
                straight_dist = np.linalg.norm([spawn_wx - goal_local[0], spawn_wy - goal_local[1]])
                if straight_dist < min_straight_dist or straight_dist >= max_straight_dist:
                    continue

                # Compute path distance using the distance map from navigation object
                distance_map = navigation.distance_maps_tensor[goal_idx].cpu().numpy()
                # Fix: Use [row, col] indexing where row=Y, col=X
                spawn_dist = distance_map[int(spawn_my), int(spawn_mx)]

                if not np.isnan(spawn_dist) and not np.isinf(spawn_dist):
                    path_dist = spawn_dist * cell_size
                    if min_path_dist <= path_dist <= max_path_dist:
                        candidates.append([spawn_wx, spawn_wy])

            # Randomly select one spawn position
            if len(candidates) > 0:
                selected_spawn = candidates[self.np_rng.integers(len(candidates))]
                spawn_positions_local.append(selected_spawn)
            else:
                spawn_positions_local.append(goal_local)

        spawn_positions_local = np.array(spawn_positions_local)

        # Transform meshes
        for mesh in temp_meshes:
            mesh.apply_translation([offset_x, offset_y, 0.0])

        # Transform to world coordinates
        rotation_matrix_2d = np.array(
            [[np.cos(base_rotation), -np.sin(base_rotation)], [np.sin(base_rotation), np.cos(base_rotation)]]
        )

        spawn_positions_world = spawn_positions_local @ rotation_matrix_2d.T
        spawn_positions_world[:, 0] += center_x
        spawn_positions_world[:, 1] += center_y

        goal_positions_world = goals_local @ rotation_matrix_2d.T
        goal_positions_world[:, 0] += center_x
        goal_positions_world[:, 1] += center_y

        for mesh in temp_meshes:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(base_rotation, [0, 0, 1]))
            mesh.apply_translation([center_x, center_y, 0.0])
            meshes_list.append(mesh)

        return spawn_positions_world, goal_positions_world, navigation

    def _create_general_nav(self, center_x, center_y, base_rotation, difficulty, meshes_list):
        """Creates a grid-based navigation environment with sub-blocks and edge walls.

        Args:
            center_x: X coordinate of the grid center
            center_y: Y coordinate of the grid center
            base_rotation: Base rotation to apply to the entire structure
            difficulty: Difficulty level (0-1), affects obstacle density
            meshes_list: List to append obstacle meshes to

        Returns:
            spawn_positions_world: numpy array of shape (N, 2) containing spawn positions
            goal_positions_world: numpy array of shape (N, 2) containing goal positions
            navigation: MazeGridNavigation object for path planning
        """
        temp_meshes = []

        def create_wall_pillars(start_pos, end_pos):
            """Creates a wall from start_pos to end_pos using small pillars."""
            wall_length = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
            if wall_length < 0.01:
                return []

            wall_direction = (np.array(end_pos) - np.array(start_pos)) / wall_length

            pillar_size = self.np_rng.uniform(0.1, 0.2)
            pillar_spacing = pillar_size * 1.2
            num_pillars = int(wall_length / pillar_spacing) + 1

            wall_pillars = []

            for i in range(num_pillars):
                pillar_offset = 0
                if self.np_rng.random() < 0.2:
                    pillar_offset = self.np_rng.uniform(0.1, 0.2)

                pillar_height = self.np_rng.uniform(0.5, 1.5)
                pillar_pos = np.array(start_pos) + wall_direction * (i * pillar_spacing)

                current_pillar_size = pillar_size * self.np_rng.uniform(0.9, 1.1)
                extents = (current_pillar_size, current_pillar_size, pillar_height)

                box = trimesh.creation.box(extents=extents)
                box.apply_translation(
                    [
                        pillar_pos[0] + wall_direction[0] * pillar_offset,
                        pillar_pos[1] + wall_direction[1] * pillar_offset,
                        pillar_height / 2.0,
                    ]
                )

                wall_pillars.append(box)

            return wall_pillars

        # Grid parameters
        cell_size = 1.25  # 2.0m per cell
        sub_block_cells = 2  # 2×2 cells per sub-block
        corridor_cells = 1  # 1 cell corridor
        num_sub_blocks = 15  # 15×15 sub-blocks

        # Calculate total grid size
        grid_size = num_sub_blocks * sub_block_cells + (num_sub_blocks - 1) * corridor_cells  # 29×29

        # Initialize maze structure (edge-based walls)
        # maze[x, y] where x is column (X-axis), y is row (Y-axis)
        maze = np.zeros((grid_size, grid_size), dtype=np.uint8)

        # Create boundary walls for the entire grid
        terrain_size = (grid_size * cell_size, grid_size * cell_size)
        meshes = create_wall_pillars((0.0, 0.0), (0.0, terrain_size[1]))
        temp_meshes.extend(meshes)
        meshes = create_wall_pillars((0.0, 0.0), (terrain_size[0], 0.0))
        temp_meshes.extend(meshes)
        meshes = create_wall_pillars((0.0, terrain_size[1]), (terrain_size[0], terrain_size[1]))
        temp_meshes.extend(meshes)
        meshes = create_wall_pillars((terrain_size[0], 0.0), (terrain_size[0], terrain_size[1]))
        temp_meshes.extend(meshes)

        # Add boundary walls to maze array
        # Left boundary (x=0)
        for y in range(grid_size):
            maze[0, y] |= 0b0001  # Left wall
        # Right boundary (x=grid_size-1)
        for y in range(grid_size):
            maze[grid_size - 1, y] |= 0b0100  # Right wall
        # Bottom boundary (y=0)
        for x in range(grid_size):
            maze[x, 0] |= 0b0010  # Bottom wall
        # Top boundary (y=grid_size-1)
        for x in range(grid_size):
            maze[x, grid_size - 1] |= 0b1000  # Top wall

        # Create walls for each sub-block
        for sb_x in range(num_sub_blocks):
            for sb_y in range(num_sub_blocks):
                # Calculate sub-block position in grid (x, y coordinates)
                start_x = sb_x * (sub_block_cells + corridor_cells)
                start_y = sb_y * (sub_block_cells + corridor_cells)

                # Randomly select 4 out of 8 segments
                while True:
                    selected_indices = self.np_rng.choice(8, size=4, replace=False)
                    # ensure no three consecutive segments are selected
                    selected_indices_sorted = np.sort(selected_indices)
                    has_three_consecutive = False
                    for i in range(8):
                        if all((selected_indices_sorted == (i + j) % 8).any() for j in range(3)):
                            has_three_consecutive = True
                            break
                    if not has_three_consecutive:
                        break

                for idx in selected_indices:
                    # Create visual walls
                    if idx == 0:  # Bottom-left horizontal segment
                        wall_start = (start_x * cell_size, start_y * cell_size)
                        wall_end = ((start_x + 1) * cell_size, start_y * cell_size)
                        # Add wall bits to maze array
                        maze[start_x, start_y] |= 0b0010  # Bottom wall of cell (x, y)
                    elif idx == 1:  # Bottom-right horizontal segment
                        wall_start = ((start_x + 1) * cell_size, start_y * cell_size)
                        wall_end = ((start_x + 2) * cell_size, start_y * cell_size)
                        # Add wall bits to maze array
                        maze[start_x + 1, start_y] |= 0b0010  # Bottom wall
                    elif idx == 2:  # Right-bottom vertical segment
                        wall_start = ((start_x + 2) * cell_size, start_y * cell_size)
                        wall_end = ((start_x + 2) * cell_size, (start_y + 1) * cell_size)
                        # Add wall bits to maze array
                        maze[start_x + 1, start_y] |= 0b0100  # Right wall
                    elif idx == 3:  # Right-top vertical segment
                        wall_start = ((start_x + 2) * cell_size, (start_y + 1) * cell_size)
                        wall_end = ((start_x + 2) * cell_size, (start_y + 2) * cell_size)
                        # Add wall bits to maze array
                        maze[start_x + 1, start_y + 1] |= 0b0100  # Right wall
                    elif idx == 4:  # Top-right horizontal segment
                        wall_start = ((start_x + 2) * cell_size, (start_y + 2) * cell_size)
                        wall_end = ((start_x + 1) * cell_size, (start_y + 2) * cell_size)
                        # Add wall bits to maze array
                        maze[start_x + 1, start_y + 1] |= 0b1000  # Top wall
                    elif idx == 5:  # Top-left horizontal segment
                        wall_start = ((start_x + 1) * cell_size, (start_y + 2) * cell_size)
                        wall_end = (start_x * cell_size, (start_y + 2) * cell_size)
                        # Add wall bits to maze array
                        maze[start_x, start_y + 1] |= 0b1000  # Top wall
                    elif idx == 6:  # Left-top vertical segment
                        wall_start = (start_x * cell_size, (start_y + 2) * cell_size)
                        wall_end = (start_x * cell_size, (start_y + 1) * cell_size)
                        # Add wall bits to maze array
                        maze[start_x, start_y + 1] |= 0b0001  # Left wall
                    else:  # idx == 7, Left-bottom vertical segment
                        wall_start = (start_x * cell_size, (start_y + 1) * cell_size)
                        wall_end = (start_x * cell_size, start_y * cell_size)
                        # Add wall bits to maze array
                        maze[start_x, start_y] |= 0b0001  # Left wall

                    temp_meshes.extend(create_wall_pillars(wall_start, wall_end))

        # Center offset
        offset_x = -(grid_size * cell_size) / 2
        offset_y = -(grid_size * cell_size) / 2

        # Calculate sub-block centers for spawn/goal sampling
        sub_block_centers = []
        for sb_x in range(num_sub_blocks):
            for sb_y in range(num_sub_blocks):
                start_x = sb_x * (sub_block_cells + corridor_cells)
                start_y = sb_y * (sub_block_cells + corridor_cells)

                # Randomly pick one of the four cells in the 2×2 sub-block
                # The four cells are at offsets (0,0), (1,0), (0,1), (1,1) from start position
                cell_offset_x = self.np_rng.integers(0, 2)  # 0 or 1
                cell_offset_y = self.np_rng.integers(0, 2)  # 0 or 1

                # Grid position of the selected cell (integer indices)
                cell_x_grid = start_x + cell_offset_x
                cell_y_grid = start_y + cell_offset_y

                # Convert to world coordinates (center of the selected cell)
                world_x = (cell_x_grid + 0.5) * cell_size
                world_y = (cell_y_grid + 0.5) * cell_size

                # Store as (x_grid, y_grid, world_x, world_y)
                sub_block_centers.append((cell_x_grid, cell_y_grid, world_x, world_y))

        sub_block_centers = np.array(sub_block_centers)

        # Generate 1000 spawn-goal pairs
        num_pairs = 1000
        spawn_positions_local = []
        goal_positions_local = []
        goal_positions_grid = []

        max_distance = 7.0  # Maximum straight-line distance in meters

        for _ in range(num_pairs):
            # Randomly select spawn position
            spawn_idx = self.np_rng.integers(len(sub_block_centers))
            spawn_x_grid, spawn_y_grid, spawn_x, spawn_y = sub_block_centers[spawn_idx]

            # Find valid goal positions within max_distance
            valid_goals = []
            for goal_idx, (goal_x_grid, goal_y_grid, goal_x, goal_y) in enumerate(sub_block_centers):
                if goal_idx == spawn_idx:
                    continue

                distance = np.linalg.norm([goal_x - spawn_x, goal_y - spawn_y])
                if distance <= max_distance:
                    valid_goals.append((goal_idx, goal_x_grid, goal_y_grid, goal_x, goal_y))

            # Select random goal from valid ones
            if len(valid_goals) > 0:
                _, goal_x_grid, goal_y_grid, goal_x, goal_y = valid_goals[self.np_rng.integers(len(valid_goals))]
            else:
                # If no valid goals, use the same position as spawn
                goal_x_grid, goal_y_grid, goal_x, goal_y = spawn_x_grid, spawn_y_grid, spawn_x, spawn_y

            spawn_positions_local.append([spawn_x, spawn_y])
            goal_positions_local.append([goal_x, goal_y])
            # Store goal positions as (x, y) grid coordinates
            goal_positions_grid.append((int(goal_x_grid), int(goal_y_grid)))

        spawn_positions_local = np.array(spawn_positions_local)
        goal_positions_local = np.array(goal_positions_local)

        # Create navigation object
        navigation = MazeGridNavigation(
            grid_size,
            grid_size,
            goal_positions_grid,
            cell_size,
            offset_x,
            offset_y,
            center_x,
            center_y,
            base_rotation,
            device=self.device,
        )
        navigation.add_walls(maze)

        # Transform meshes to centered coordinates
        for mesh in temp_meshes:
            mesh.apply_translation([offset_x, offset_y, 0.0])

        # Transform spawn/goal positions to centered local coordinates
        spawn_positions_local[:, 0] += offset_x
        spawn_positions_local[:, 1] += offset_y
        goal_positions_local[:, 0] += offset_x
        goal_positions_local[:, 1] += offset_y

        # Transform to world coordinates with rotation
        rotation_matrix_2d = np.array(
            [[np.cos(base_rotation), -np.sin(base_rotation)], [np.sin(base_rotation), np.cos(base_rotation)]]
        )

        spawn_positions_world = spawn_positions_local @ rotation_matrix_2d.T
        spawn_positions_world[:, 0] += center_x
        spawn_positions_world[:, 1] += center_y

        goal_positions_world = goal_positions_local @ rotation_matrix_2d.T
        goal_positions_world[:, 0] += center_x
        goal_positions_world[:, 1] += center_y

        # Apply rotation and translation to meshes
        for mesh in temp_meshes:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(base_rotation, [0, 0, 1]))
            mesh.apply_translation([center_x, center_y, 0.0])
            meshes_list.append(mesh)

        return spawn_positions_world, goal_positions_world, navigation

    def _create_dead_end_nav(self, center_x, center_y, base_rotation, difficulty, meshes_list):
        temp_meshes = []
        spawn_positions_local = []
        goal_positions_local = []

        # Grid parameters
        cell_size = 0.25
        grid_rows = 320
        grid_cols = 320
        block_size = 40
        block_rows = grid_rows // block_size
        block_cols = grid_cols // block_size

        # Initialize occupancy map
        occupancy_map = np.zeros((grid_rows, grid_cols), dtype=bool)

        # Store dead-end centers and their exclusion radii for obstacle placement
        dead_end_exclusions = []

        def create_wall_pillars(start_pos, end_pos):
            """Creates a wall from start_pos to end_pos using small pillars."""
            wall_length = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
            if wall_length < 0.01:
                return []

            wall_direction = (np.array(end_pos) - np.array(start_pos)) / wall_length

            pillar_size = np.random.uniform(0.1, 0.2)
            pillar_spacing = pillar_size * 1.2
            num_pillars = int(wall_length / pillar_spacing) + 1

            wall_pillars = []

            for i in range(num_pillars):
                # Randomly create gaps (max 0.2m)
                pillar_offset = 0
                if np.random.random() < 0.2:  # 20% chance of gap
                    pillar_offset = np.random.uniform(0.1, 0.2)

                pillar_height = np.random.uniform(0.5, 1.5)
                pillar_pos = np.array(start_pos) + wall_direction * (i * pillar_spacing)

                current_pillar_size = pillar_size * np.random.uniform(0.9, 1.1)
                extents = (current_pillar_size, current_pillar_size, pillar_height)

                box = trimesh.creation.box(extents=extents)
                box.apply_translation(
                    [
                        pillar_pos[0] + wall_direction[0] * pillar_offset,
                        pillar_pos[1] + wall_direction[1] * pillar_offset,
                        pillar_height / 2.0,
                    ]
                )

                wall_pillars.append(box)

            return wall_pillars

        def create_2x2_pillar_grid(grid_x, grid_y, cell_size):
            """Creates a 2x2 grid of square pillars spanning four neighboring cells.

            Args:
                grid_x: Grid x coordinate of bottom-left cell
                grid_y: Grid y coordinate of bottom-left cell
                cell_size: Size of each grid cell

            Returns:
                List of pillar meshes
            """
            pillars = []
            pillar_size = np.random.uniform(0.15, 0.25)
            pillar_height = np.random.uniform(0.5, 1.5)

            # Create 2x2 grid of pillars
            for i in range(2):
                for j in range(2):
                    pillar_x = (grid_x + i) * cell_size + cell_size / 2
                    pillar_y = (grid_y + j) * cell_size + cell_size / 2

                    # Add small random offset
                    pillar_x += np.random.uniform(-cell_size * 0.2, cell_size * 0.2)
                    pillar_y += np.random.uniform(-cell_size * 0.2, cell_size * 0.2)

                    extents = (pillar_size, pillar_size, pillar_height)
                    box = trimesh.creation.box(extents=extents)
                    box.apply_translation([pillar_x, pillar_y, pillar_height / 2.0])
                    pillars.append(box)

            return pillars

        def create_dead_end_wall(center_x, center_y, central_wall_length, side_wall_length):
            """Creates a U-shaped dead-end wall (3 sides) with random orientation.
            Places walls inside cells rather than along edges.
            The entire U-shape is centered at (center_x, center_y).
            Returns (walls, orientation, spawn_pos, goal_pos, occupied_cells)
            """
            orientation = np.random.randint(0, 4)
            walls = []
            spawn_pos = None
            goal_pos = None
            occupied_cells = []

            # Convert wall dimensions to cell units
            central_cells = int(central_wall_length / cell_size)
            side_cells = int(side_wall_length / cell_size)

            if orientation == 0:  # Opening at bottom
                # Center the central wall horizontally around center_x
                start_col = int(center_x / cell_size) - central_cells // 2
                # Position central wall at center_y + side_cells/2 to center the U-shape
                central_row = int(center_y / cell_size) + side_cells // 2

                # Central wall (horizontal)
                for i in range(central_cells):
                    col = start_col + i
                    if 0 <= col < grid_cols and 0 <= central_row < grid_rows:
                        occupied_cells.append((central_row, col))
                        pillar_x = col * cell_size + cell_size / 2
                        pillar_y = central_row * cell_size + cell_size / 2
                        walls.extend(create_wall_pillars((pillar_x, pillar_y), (pillar_x + 0.1, pillar_y + 0.1)))

                # Left side wall (vertical)
                for i in range(side_cells):
                    row = central_row - i - 1
                    if 0 <= start_col < grid_cols and 0 <= row < grid_rows:
                        occupied_cells.append((row, start_col))
                        pillar_x = start_col * cell_size + cell_size / 2
                        pillar_y = row * cell_size + cell_size / 2
                        walls.extend(create_wall_pillars((pillar_x, pillar_y), (pillar_x + 0.1, pillar_y + 0.1)))

                # Right side wall (vertical)
                right_col = start_col + central_cells - 1
                for i in range(side_cells):
                    row = central_row - i - 1
                    if 0 <= right_col < grid_cols and 0 <= row < grid_rows:
                        occupied_cells.append((row, right_col))
                        pillar_x = right_col * cell_size + cell_size / 2
                        pillar_y = row * cell_size + cell_size / 2
                        walls.extend(create_wall_pillars((pillar_x, pillar_y), (pillar_x + 0.1, pillar_y + 0.1)))

                # Spawn inside at the back (top), goal outside beyond the back wall
                spawn_pos = [center_x, center_y + (side_cells * cell_size / 2 - 1.0)]
                goal_pos = [center_x, center_y + side_cells * cell_size / 2 + 1.5]

            elif orientation == 1:  # Opening at right
                # Center the central wall vertically around center_y
                start_row = int(center_y / cell_size) - central_cells // 2
                # Position central wall at center_x - side_cells/2 to center the U-shape
                central_col = int(center_x / cell_size) - side_cells // 2

                # Central wall (vertical)
                for i in range(central_cells):
                    row = start_row + i
                    if 0 <= central_col < grid_cols and 0 <= row < grid_rows:
                        occupied_cells.append((row, central_col))
                        pillar_x = central_col * cell_size + cell_size / 2
                        pillar_y = row * cell_size + cell_size / 2
                        walls.extend(create_wall_pillars((pillar_x, pillar_y), (pillar_x + 0.1, pillar_y + 0.1)))

                # Top and bottom side walls (horizontal)
                top_row = start_row + central_cells - 1
                bottom_row = start_row
                for i in range(side_cells):
                    col = central_col + i + 1
                    if 0 <= col < grid_cols and 0 <= top_row < grid_rows:
                        occupied_cells.append((top_row, col))
                        pillar_x = col * cell_size + cell_size / 2
                        pillar_y = top_row * cell_size + cell_size / 2
                        walls.extend(create_wall_pillars((pillar_x, pillar_y), (pillar_x + 0.1, pillar_y + 0.1)))
                    if 0 <= col < grid_cols and 0 <= bottom_row < grid_rows:
                        occupied_cells.append((bottom_row, col))
                        pillar_x = col * cell_size + cell_size / 2
                        pillar_y = bottom_row * cell_size + cell_size / 2
                        walls.extend(create_wall_pillars((pillar_x, pillar_y), (pillar_x + 0.1, pillar_y + 0.1)))

                # Spawn inside at the back (left), goal outside beyond the back wall
                spawn_pos = [center_x - (side_cells * cell_size / 2 - 1.0), center_y]
                goal_pos = [center_x - side_cells * cell_size / 2 - 1.5, center_y]

            elif orientation == 2:  # Opening at top
                # Center the central wall horizontally around center_x
                start_col = int(center_x / cell_size) - central_cells // 2
                # Position central wall at center_y - side_cells/2 to center the U-shape
                central_row = int(center_y / cell_size) - side_cells // 2

                # Central wall (horizontal)
                for i in range(central_cells):
                    col = start_col + i
                    if 0 <= col < grid_cols and 0 <= central_row < grid_rows:
                        occupied_cells.append((central_row, col))
                        pillar_x = col * cell_size + cell_size / 2
                        pillar_y = central_row * cell_size + cell_size / 2
                        walls.extend(create_wall_pillars((pillar_x, pillar_y), (pillar_x + 0.1, pillar_y + 0.1)))

                # Left side wall (vertical)
                for i in range(side_cells):
                    row = central_row + i + 1
                    if 0 <= start_col < grid_cols and 0 <= row < grid_rows:
                        occupied_cells.append((row, start_col))
                        pillar_x = start_col * cell_size + cell_size / 2
                        pillar_y = row * cell_size + cell_size / 2
                        walls.extend(create_wall_pillars((pillar_x, pillar_y), (pillar_x + 0.1, pillar_y + 0.1)))

                # Right side wall (vertical)
                right_col = start_col + central_cells - 1
                for i in range(side_cells):
                    row = central_row + i + 1
                    if 0 <= right_col < grid_cols and 0 <= row < grid_rows:
                        occupied_cells.append((row, right_col))
                        pillar_x = right_col * cell_size + cell_size / 2
                        pillar_y = row * cell_size + cell_size / 2
                        walls.extend(create_wall_pillars((pillar_x, pillar_y), (pillar_x + 0.1, pillar_y + 0.1)))

                # Spawn inside at the back (bottom), goal outside beyond the back wall
                spawn_pos = [center_x, center_y - (side_cells * cell_size / 2 - 1.0)]
                goal_pos = [center_x, center_y - side_cells * cell_size / 2 - 1.5]

            else:  # orientation == 3, Opening at left
                # Center the central wall vertically around center_y
                start_row = int(center_y / cell_size) - central_cells // 2
                # Position central wall at center_x + side_cells/2 to center the U-shape
                central_col = int(center_x / cell_size) + side_cells // 2

                # Central wall (vertical)
                for i in range(central_cells):
                    row = start_row + i
                    if 0 <= central_col < grid_cols and 0 <= row < grid_rows:
                        occupied_cells.append((row, central_col))
                        pillar_x = central_col * cell_size + cell_size / 2
                        pillar_y = row * cell_size + cell_size / 2
                        walls.extend(create_wall_pillars((pillar_x, pillar_y), (pillar_x + 0.1, pillar_y + 0.1)))

                # Top and bottom side walls (horizontal)
                top_row = start_row + central_cells - 1
                bottom_row = start_row
                for i in range(side_cells):
                    col = central_col - i - 1
                    if 0 <= col < grid_cols and 0 <= top_row < grid_rows:
                        occupied_cells.append((top_row, col))
                        pillar_x = col * cell_size + cell_size / 2
                        pillar_y = top_row * cell_size + cell_size / 2
                        walls.extend(create_wall_pillars((pillar_x, pillar_y), (pillar_x + 0.1, pillar_y + 0.1)))
                    if 0 <= col < grid_cols and 0 <= bottom_row < grid_rows:
                        occupied_cells.append((bottom_row, col))
                        pillar_x = col * cell_size + cell_size / 2
                        pillar_y = bottom_row * cell_size + cell_size / 2
                        walls.extend(create_wall_pillars((pillar_x, pillar_y), (pillar_x + 0.1, pillar_y + 0.1)))

                # Spawn inside at the back (right), goal outside beyond the back wall
                spawn_pos = [center_x + (side_cells * cell_size / 2 - 1.0), center_y]
                goal_pos = [center_x + side_cells * cell_size / 2 + 1.5, center_y]

            return walls, orientation, spawn_pos, goal_pos, occupied_cells

        # Create boundary walls (mark boundary cells as occupied)
        border = 4 * cell_size
        # Create visual boundary walls
        meshes = create_wall_pillars(
            (-border, -border),
            (-border, grid_rows * cell_size + border),
        )
        temp_meshes.extend(meshes)
        meshes = create_wall_pillars(
            (-border, -border),
            (grid_cols * cell_size + border, -border),
        )
        temp_meshes.extend(meshes)
        meshes = create_wall_pillars(
            (-border, grid_rows * cell_size + border),
            (grid_cols * cell_size + border, grid_rows * cell_size + border),
        )
        temp_meshes.extend(meshes)
        meshes = create_wall_pillars(
            (grid_cols * cell_size + border, -border),
            (grid_cols * cell_size + border, grid_rows * cell_size + border),
        )
        temp_meshes.extend(meshes)

        # Mark boundary cells as occupied in occupancy map
        for i in range(grid_cols):
            occupancy_map[0, i] = True  # Top boundary
            occupancy_map[grid_rows - 1, i] = True  # Bottom boundary
        for i in range(grid_rows):
            occupancy_map[i, 0] = True  # Left boundary
            occupancy_map[i, grid_cols - 1] = True  # Right boundary

        # Create dead end scenarios at the center of each block
        for i in range(block_rows):
            for j in range(block_cols):
                block_center_x = (i * block_size + block_size // 2) * cell_size
                block_center_y = (j * block_size + block_size // 2) * cell_size

                central_wall_length = np.random.randint(7, 12) * cell_size
                side_wall_length = np.random.randint(1, 12) * cell_size

                dead_end_walls, orientation, spawn_pos, goal_pos, occupied_cells = create_dead_end_wall(
                    block_center_x, block_center_y, central_wall_length, side_wall_length
                )
                temp_meshes.extend(dead_end_walls)

                # Mark occupied cells in occupancy map
                for row, col in occupied_cells:
                    if 0 <= row < grid_rows and 0 <= col < grid_cols:
                        occupancy_map[row, col] = True

                spawn_positions_local.append(spawn_pos)
                goal_positions_local.append(goal_pos)

                # Calculate exclusion radius for this dead-end
                # Use the larger of the two wall dimensions plus a buffer
                exclusion_radius = max(central_wall_length, side_wall_length) / 2 + 1.0  # +1m buffer
                dead_end_exclusions.append((block_center_x, block_center_y, exclusion_radius))

        # Add randomly scattered 2x2 pillar grids
        num_pillar_grids = 200
        exclusion_radius_spawn_goal = 2.0

        for _ in range(num_pillar_grids):
            for attempt in range(100):
                # Generate grid coordinates (row, col) - be consistent with occupancy_map indexing
                grid_row = np.random.randint(0, grid_rows - 1)
                grid_col = np.random.randint(0, grid_cols - 1)

                # Convert grid indices to world coordinates for distance checking
                pillar_world_x = grid_col * cell_size  # col corresponds to x
                pillar_world_y = grid_row * cell_size  # row corresponds to y

                # Check distance to spawn positions
                too_close = False
                for spawn_pos in spawn_positions_local:
                    if (
                        np.linalg.norm([pillar_world_x - spawn_pos[0], pillar_world_y - spawn_pos[1]])
                        < exclusion_radius_spawn_goal
                    ):
                        too_close = True
                        break

                if not too_close:
                    # Check distance to goal positions
                    for goal_pos in goal_positions_local:
                        if (
                            np.linalg.norm([pillar_world_x - goal_pos[0], pillar_world_y - goal_pos[1]])
                            < exclusion_radius_spawn_goal
                        ):
                            too_close = True
                            break

                if not too_close:
                    # NEW CHECK: Check distance to dead-end centers
                    for dead_end_x, dead_end_y, exclusion_radius in dead_end_exclusions:
                        if (
                            np.linalg.norm([pillar_world_x - dead_end_x, pillar_world_y - dead_end_y])
                            < exclusion_radius
                        ):
                            too_close = True
                            break

                if not too_close:
                    # Create pillars using the correct grid coordinates
                    pillar_grid = create_2x2_pillar_grid(grid_col, grid_row, cell_size)  # (x, y) = (col, row)
                    temp_meshes.extend(pillar_grid)

                    # Mark 2x2 cells as occupied in occupancy map using (row, col) indexing
                    for pi in range(2):
                        for pj in range(2):
                            if 0 <= grid_row + pi < grid_rows and 0 <= grid_col + pj < grid_cols:
                                occupancy_map[grid_row + pi, grid_col + pj] = True
                    break

        # Center the entire grid
        offset_x = -(grid_cols * cell_size) / 2
        offset_y = -(grid_rows * cell_size) / 2

        for mesh in temp_meshes:
            mesh.apply_translation([offset_x, offset_y, 0.0])

        # Transform spawn positions to centered local coordinates
        spawn_positions_local = np.array(spawn_positions_local)
        spawn_positions_local[:, 0] += offset_x
        spawn_positions_local[:, 1] += offset_y

        # Transform goal positions to centered local coordinates
        goal_positions_local = np.array(goal_positions_local)
        goal_positions_local[:, 0] += offset_x
        goal_positions_local[:, 1] += offset_y

        # Transform to world coordinates
        rotation_matrix_2d = np.array(
            [[np.cos(base_rotation), -np.sin(base_rotation)], [np.sin(base_rotation), np.cos(base_rotation)]]
        )

        # Rotate spawn positions
        spawn_positions_world = spawn_positions_local @ rotation_matrix_2d.T
        spawn_positions_world[:, 0] += center_x
        spawn_positions_world[:, 1] += center_y

        # Rotate goal positions
        goal_positions_world = goal_positions_local @ rotation_matrix_2d.T
        goal_positions_world[:, 0] += center_x
        goal_positions_world[:, 1] += center_y

        for mesh in temp_meshes:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(base_rotation, [0, 0, 1]))
            mesh.apply_translation([center_x, center_y, 0.0])
            meshes_list.append(mesh)

        # Convert goal positions to grid indices
        goal_positions_grid = []
        for goal_pos in goal_positions_local:
            # goal_pos is already in local coordinates (after offset adjustment)
            goal_grid_x = goal_pos[0] - offset_x
            goal_grid_y = goal_pos[1] - offset_y
            goal_col = int(goal_grid_x / cell_size)
            goal_row = int(goal_grid_y / cell_size)

            # Clamp to valid grid bounds
            goal_col = max(0, min(goal_col, grid_cols - 1))
            goal_row = max(0, min(goal_row, grid_rows - 1))

            # Ensure the goal cell is not occupied
            if occupancy_map[goal_row, goal_col]:
                # Find nearest unoccupied cell
                found_free = False
                for radius in range(1, 10):
                    for dr in range(-radius, radius + 1):
                        for dc in range(-radius, radius + 1):
                            new_row = goal_row + dr
                            new_col = goal_col + dc
                            if (
                                0 <= new_row < grid_rows
                                and 0 <= new_col < grid_cols
                                and not occupancy_map[new_row, new_col]
                            ):
                                goal_row, goal_col = new_row, new_col
                                found_free = True
                                break
                        if found_free:
                            break
                    if found_free:
                        break

            goal_positions_grid.append((goal_row, goal_col))

        spawn_positions_world = spawn_positions_world[-5:]
        goal_positions_world = goal_positions_world[-5:]
        goal_positions_grid = goal_positions_grid[-5:]

        # Create navigation object
        navigation = GridMapNavigation(
            occupancy_map, goal_positions_grid, cell_size, offset_x, offset_y, center_x, center_y, base_rotation
        )

        return spawn_positions_world, goal_positions_world, navigation
