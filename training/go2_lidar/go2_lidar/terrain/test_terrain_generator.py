# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import datetime
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

if TYPE_CHECKING:
    from isaaclab.terrains.sub_terrain_cfg import FlatPatchSamplingCfg, SubTerrainBaseCfg
    from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


class TestTerrainGenerator:
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
        if self.cfg.num_rows != 10 or self.cfg.num_cols != 8:
            raise ValueError(
                "Curriculum obstacles are only supported for 10 rows and 8 columns. "
                f"Current configuration has {self.cfg.num_rows} rows and {self.cfg.num_cols} columns."
            )

        num_test_cases = 8
        num_variants_per_case = self.cfg.num_rows
        terrain_size = (self.cfg.size[0] * num_variants_per_case, self.cfg.size[1] * num_test_cases)
        cell_size_x = self.cfg.size[0]
        cell_size_y = self.cfg.size[1]

        """1: Scatter-Static"""
        np.random.seed(datetime.datetime.now().microsecond)
        centers = np.zeros((num_variants_per_case, 2))
        centers[:, 0] = np.arange(num_variants_per_case) * cell_size_x + cell_size_x / 2.0
        centers[:, 1] = cell_size_y / 2.0
        for c in centers:
            self.create_env_scatter_static(self.terrain_meshes, c[0], c[1], cell_size_x * 0.75, cell_size_y * 0.75)
        starts = torch.from_numpy(centers).to(self.device).to(torch.float)
        starts[:, 0] -= self.cfg.size[0] * 0.75 / 2.0 - 2.0
        goals = torch.from_numpy(centers).to(self.device).to(torch.float)
        goals[:, 0] += self.cfg.size[0] * 0.75 / 2.0 - 2.0
        self.flat_patches["scatter_static_start"] = starts.clone()
        self.flat_patches["scatter_static_goal"] = goals.clone()

        """2: Scatter-Dynamic"""
        np.random.seed(datetime.datetime.now().microsecond)
        centers[:, 1] = cell_size_y * 1.0 + cell_size_y / 2.0
        for c in centers:
            self.create_env_scatter_dynamic(self.terrain_meshes, c[0], c[1], cell_size_x * 0.75, cell_size_y * 0.75)
        starts = torch.from_numpy(centers).to(self.device).to(torch.float)
        starts[:, 0] -= self.cfg.size[0] * 0.75 / 2.0 - 2.0
        goals = torch.from_numpy(centers).to(self.device).to(torch.float)
        goals[:, 0] += self.cfg.size[0] * 0.75 / 2.0 - 2.0
        self.flat_patches["scatter_dynamic_start"] = starts.clone()
        self.flat_patches["scatter_dynamic_goal"] = goals.clone()

        """3: Return"""
        np.random.seed(datetime.datetime.now().microsecond)
        centers[:, 1] = cell_size_y * 2.0 + cell_size_y / 2.0
        for c in centers:
            self.create_env_return(self.terrain_meshes, c[0], c[1], cell_size_x * 0.75, cell_size_y * 0.75)
        starts = torch.from_numpy(centers).to(self.device).to(torch.float)
        self.flat_patches["return_start"] = starts.clone()
        self.flat_patches["return_goal"] = starts.clone()

        """4: Scatter-Static-Sparse"""
        np.random.seed(datetime.datetime.now().microsecond)
        centers[:, 1] = cell_size_y * 3.0 + cell_size_y / 2.0
        for c in centers:
            self.create_env_scatter_static_sparse(
                self.terrain_meshes, c[0], c[1], cell_size_x * 0.75, cell_size_y * 0.75
            )
        starts = torch.from_numpy(centers).to(self.device).to(torch.float)
        starts[:, 0] -= self.cfg.size[0] * 0.75 / 2.0 - 2.0
        goals = torch.from_numpy(centers).to(self.device).to(torch.float)
        goals[:, 0] += self.cfg.size[0] * 0.75 / 2.0 - 2.0
        self.flat_patches["scatter_static_sparse_start"] = starts.clone()
        self.flat_patches["scatter_static_sparse_goal"] = goals.clone()

        """5: Scatter-Static-Dense"""
        np.random.seed(datetime.datetime.now().microsecond)
        centers[:, 1] = cell_size_y * 4.0 + cell_size_y / 2.0
        for c in centers:
            self.create_env_scatter_static_dense(
                self.terrain_meshes, c[0], c[1], cell_size_x * 0.75, cell_size_y * 0.75
            )
        starts = torch.from_numpy(centers).to(self.device).to(torch.float)
        starts[:, 0] -= self.cfg.size[0] * 0.75 / 2.0 - 2.0
        goals = torch.from_numpy(centers).to(self.device).to(torch.float)
        goals[:, 0] += self.cfg.size[0] * 0.75 / 2.0 - 2.0
        self.flat_patches["scatter_static_dense_start"] = starts.clone()
        self.flat_patches["scatter_static_dense_goal"] = goals.clone()

        """6: Maze"""
        # Define maze layout using string representation
        maze_layouts = [
            """
            d x r x x d x d r x
            x r x d x r d r x d
            gx r d r r x x x r xs
            rd d x d x d x dr x d
            x x r x r x x x x x
            """,
            """
            r x r x d x d x x x
            d r d r x d l x x d
            gx d d x x d x d l xs
            r x r x x d r x x x
            x r x r r x x r r x
            """,
            """
            d x r d r x d d x d
            x r x d x x r x x x
            gdr d r r x dr x d x xs
            x d r x d x x x dr x
            r x x r x r x x r x
            """,
            """
            x r r x r x x dr x x
            d x d r x r d x d x
            gx r d x dr r r x d xs
            r d x r d x d r d x
            x x r x x x x x x x
            """,
            """
            d x x d d x r r x x
            d r x r x r x d d x
            gd x dr d dr r x x d xs
            x r x x r x r x dr x
            x x r x x r x x x x
            """,
            """
            d r d x r x d r d d
            x d x r x d r x x x
            gr x x d d dr x x x xs
            x dr x x d r x dr r x
            x x r r x x x x r x
            """,
            """
            d r x r x r r d x x
            d x r x rd x x d x x
            gd r x d r r r x r xs
            r r d r x d x r x d
            x x x x r x r x x x
            """,
            """
            r d x x d r d x d x
            d x r r d x d x x d
            gr x rd x x r x d x ds
            x x d r dr r x dr r x
            x r x r x x x x x x
            """,
            """
            d x d r x d d x d x
            r x d x d r x r r x
            gr x r x r d r x x xs
            d r d r x x d r r d
            x x x x r x x r x x
            """,
            """
            x r x r x d d x x l
            r d r d d r x x d x
            g r x r x d d r x s
            r x r d x d r x d x
            x r r x x x x r x x
            """,
        ]
        centers[:, 1] = cell_size_y * 5.0 + cell_size_y / 2.0
        starts = centers.copy()
        goals = centers.copy()
        for i, c in enumerate(centers):
            start_pos, goal_pos, _ = self.create_env_dynamic_maze(
                self.terrain_meshes, c[0], c[1], cell_size_x, cell_size_y, maze_layouts[i]
            )
            starts[i, :] = start_pos
            goals[i, :] = goal_pos
        self.flat_patches["maze_start"] = torch.from_numpy(starts).to(self.device).to(torch.float).clone()
        self.flat_patches["maze_goal"] = torch.from_numpy(goals).to(self.device).to(torch.float).clone()

        """7: Maze-Dynamic"""
        centers[:, 1] = cell_size_y * 6.0 + cell_size_y / 2.0
        starts = centers.copy()
        goals = centers.copy()
        for i, c in enumerate(centers):
            start_pos, goal_pos, _ = self.create_env_dynamic_maze(
                self.terrain_meshes, c[0], c[1], cell_size_x, cell_size_y, maze_layouts[i]
            )
            starts[i, :] = start_pos
            goals[i, :] = goal_pos
        self.flat_patches["dynamic_maze_start"] = torch.from_numpy(starts).to(self.device).to(torch.float).clone()
        self.flat_patches["dynamic_maze_goal"] = torch.from_numpy(goals).to(self.device).to(torch.float).clone()

        """8: Scatter-Dynamic-Sparse"""
        np.random.seed(datetime.datetime.now().microsecond)
        centers[:, 1] = cell_size_y * 7.0 + cell_size_y / 2.0
        for c in centers:
            self.create_env_scatter_dynamic_sparse(
                self.terrain_meshes, c[0], c[1], cell_size_x * 0.75, cell_size_y * 0.75
            )
        starts = torch.from_numpy(centers).to(self.device).to(torch.float)
        starts[:, 0] -= self.cfg.size[0] * 0.75 / 2.0 - 2.0
        goals = torch.from_numpy(centers).to(self.device).to(torch.float)
        goals[:, 0] += self.cfg.size[0] * 0.75 / 2.0 - 2.0
        self.flat_patches["scatter_dynamic_sparse_start"] = starts.clone()
        self.flat_patches["scatter_dynamic_sparse_goal"] = goals.clone()

    def create_jagged_wall(self, meshes_list, start_pos, end_pos, pillar_size_range=None):
        """Creates a jagged wall from start_pos to end_pos using varied square pillars."""
        wall_length = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
        if wall_length < 0.01:
            return

        wall_direction = (np.array(end_pos) - np.array(start_pos)) / wall_length

        # Vary pillar parameters
        current_pos = 0
        while current_pos < wall_length:
            # Random pillar size (thick/thin)
            pillar_size = np.random.uniform(0.15, 0.4)
            if pillar_size_range is not None:
                pillar_size = np.random.uniform(pillar_size_range[0], pillar_size_range[1])
            # Random height (high/low)
            pillar_height = np.random.uniform(0.3, 2.0)
            # Random spacing between pillars
            spacing = np.random.uniform(0, 0.15)

            pillar_pos = np.array(start_pos) + wall_direction * current_pos

            extents = (pillar_size, pillar_size, pillar_height)
            box = trimesh.creation.box(extents=extents)
            box.apply_translation([pillar_pos[0], pillar_pos[1], pillar_height / 2.0])
            meshes_list.append(box)

            current_pos += pillar_size + spacing

    def create_composite_obstacle(self, center_x, center_y, base_rotation, meshes_list):
        """Creates a composite obstacle made of 3-4 smaller boxes."""
        num_boxes = np.random.randint(3, 5)  # 3 or 4 boxes
        pattern = np.random.choice(["row", "L_shape", "grid_2x2", "T_shape"])

        # Configurable parameters
        box_size_range = (0.1, 0.3)
        base_size = np.random.uniform(*box_size_range)
        height_range = (0.5, 1.5)  # (min_height, max_height) - easily configurable

        # Random orientation for the entire composite obstacle
        composite_rotation = np.random.uniform(0, 2 * np.pi)

        # Store boxes to apply composite rotation later
        temp_boxes = []

        if pattern == "row":
            # Boxes in a line
            for i in range(num_boxes):
                box_size = base_size * np.random.uniform(0.8, 1.2)
                box_height = np.random.uniform(height_range[0], height_range[1])
                offset_x = (i - (num_boxes - 1) / 2) * box_size * 1.1
                offset_y = 0

                extents = (box_size, box_size, box_height)
                pos = [offset_x, offset_y, box_height / 2.0]

                box = trimesh.creation.box(extents=extents)
                box.apply_transform(trimesh.transformations.rotation_matrix(base_rotation, [0, 0, 1]))
                box.apply_translation(pos)
                temp_boxes.append((box, extents, pos))

        elif pattern == "L_shape":
            # L-shaped pattern
            positions = [(0, 0), (1, 0), (0, 1)] if num_boxes == 3 else [(0, 0), (1, 0), (0, 1), (0, 2)]
            for i, (dx, dy) in enumerate(positions):
                box_size = base_size * np.random.uniform(0.8, 1.2)
                box_height = np.random.uniform(height_range[0], height_range[1])
                offset_x = dx * box_size * 1.1 - box_size * 0.5
                offset_y = dy * box_size * 1.1 - box_size * 0.5

                extents = (box_size, box_size, box_height)
                pos = [offset_x, offset_y, box_height / 2.0]

                box = trimesh.creation.box(extents=extents)
                box.apply_transform(trimesh.transformations.rotation_matrix(base_rotation, [0, 0, 1]))
                box.apply_translation(pos)
                temp_boxes.append((box, extents, pos))

        elif pattern == "grid_2x2":
            # 2x2 grid (only for 4 boxes)
            if num_boxes == 4:
                positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
            else:
                positions = [(0, 0), (1, 0), (0, 1)]  # 3 boxes in partial grid

            for i, (dx, dy) in enumerate(positions):
                box_size = base_size * np.random.uniform(0.8, 1.2)
                box_height = np.random.uniform(height_range[0], height_range[1])
                offset_x = dx * box_size * 1.1 - box_size * 0.5
                offset_y = dy * box_size * 1.1 - box_size * 0.5

                extents = (box_size, box_size, box_height)
                pos = [offset_x, offset_y, box_height / 2.0]

                box = trimesh.creation.box(extents=extents)
                box.apply_transform(trimesh.transformations.rotation_matrix(base_rotation, [0, 0, 1]))
                box.apply_translation(pos)
                temp_boxes.append((box, extents, pos))

        elif pattern == "T_shape":
            # T-shaped pattern
            if num_boxes == 3:
                positions = [(0, 0), (-1, 1), (1, 1)]
            else:
                positions = [(0, 0), (-1, 1), (0, 1), (1, 1)]

            for i, (dx, dy) in enumerate(positions):
                box_size = base_size * np.random.uniform(0.8, 1.2)
                box_height = np.random.uniform(height_range[0], height_range[1])
                offset_x = dx * box_size * 1.1
                offset_y = dy * box_size * 1.1

                extents = (box_size, box_size, box_height)
                pos = [offset_x, offset_y, box_height / 2.0]

                box = trimesh.creation.box(extents=extents)
                box.apply_transform(trimesh.transformations.rotation_matrix(base_rotation, [0, 0, 1]))
                box.apply_translation(pos)
                temp_boxes.append((box, extents, pos))

        # Apply composite rotation and final translation to all boxes
        # center_x += np.random.uniform(-0.5, 0.5)
        # center_y += np.random.uniform(-0.5, 0.5)
        for box, extents, local_pos in temp_boxes:
            # Apply composite rotation around origin
            box.apply_transform(trimesh.transformations.rotation_matrix(composite_rotation, [0, 0, 1]))
            # Translate to final position
            box.apply_translation([center_x, center_y, 0.0])

            # Calculate final box center position for collision check
            # final_box_center = [center_x + local_pos[0], center_y + local_pos[1]]

            # Only add the box if it doesn't intersect with the center exclusion zone
            # if not _is_box_in_center_zone(final_box_center, extents):
            meshes_list.append(box)

    def create_simple_composite_obstacle(self, center_x, center_y, base_rotation, meshes_list, box_size_range=None):
        """Creates a composite obstacle made of 3-4 smaller boxes."""
        num_boxes = np.random.randint(3, 5)  # 3 or 4 boxes

        # Configurable parameters
        if box_size_range is None:
            box_size_range = (0.1, 0.2)
        base_size = np.random.uniform(*box_size_range)
        height_range = (0.5, 1.5)  # (min_height, max_height) - easily configurable

        # Random orientation for the entire composite obstacle
        composite_rotation = np.random.uniform(0, 2 * np.pi)

        # Store boxes to apply composite rotation later
        temp_boxes = []

        # 2x2 grid (only for 4 boxes)
        if num_boxes == 4:
            positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
        else:
            positions = [(0, 0), (1, 0), (0, 1)]  # 3 boxes in partial grid

        for i, (dx, dy) in enumerate(positions):
            box_size = base_size * np.random.uniform(0.8, 1.2)
            box_height = np.random.uniform(height_range[0], height_range[1])
            offset_x = dx * box_size * 1.1 - box_size * 0.5
            offset_y = dy * box_size * 1.1 - box_size * 0.5

            extents = (box_size, box_size, box_height)
            pos = [offset_x, offset_y, box_height / 2.0]

            box = trimesh.creation.box(extents=extents)
            box.apply_transform(trimesh.transformations.rotation_matrix(base_rotation, [0, 0, 1]))
            box.apply_translation(pos)
            temp_boxes.append((box, extents, pos))

        # Apply composite rotation and final translation to all boxes
        # center_x += np.random.uniform(-0.5, 0.5)
        # center_y += np.random.uniform(-0.5, 0.5)
        for box, extents, local_pos in temp_boxes:
            # Apply composite rotation around origin
            box.apply_transform(trimesh.transformations.rotation_matrix(composite_rotation, [0, 0, 1]))
            # Translate to final position
            box.apply_translation([center_x, center_y, 0.0])

            # Calculate final box center position for collision check
            # final_box_center = [center_x + local_pos[0], center_y + local_pos[1]]

            # Only add the box if it doesn't intersect with the center exclusion zone
            # if not _is_box_in_center_zone(final_box_center, extents):
            meshes_list.append(box)

    def create_enclosing_walls(self, meshes_list, center_x, center_y, size_x, size_y):
        terrain_size = (size_x, size_y)
        # Left wall
        self.create_jagged_wall(
            meshes_list,
            (center_x - terrain_size[0] / 2, center_y - terrain_size[1] / 2),
            (center_x - terrain_size[0] / 2, center_y + terrain_size[1] / 2),
        )
        # Right wall
        self.create_jagged_wall(
            meshes_list,
            (center_x + terrain_size[0] / 2, center_y - terrain_size[1] / 2),
            (center_x + terrain_size[0] / 2, center_y + terrain_size[1] / 2),
        )
        # Bottom wall
        self.create_jagged_wall(
            meshes_list,
            (center_x - terrain_size[0] / 2, center_y - terrain_size[1] / 2),
            (center_x + terrain_size[0] / 2, center_y - terrain_size[1] / 2),
        )
        # Top wall
        self.create_jagged_wall(
            meshes_list,
            (center_x - terrain_size[0] / 2, center_y + terrain_size[1] / 2),
            (center_x + terrain_size[0] / 2, center_y + terrain_size[1] / 2),
        )

    def create_env_scatter_static(self, meshes_list, center_x, center_y, size_x, size_y):
        terrain_size = (size_x, size_y)

        # Create enclosing walls
        self.create_enclosing_walls(meshes_list, center_x, center_y, size_x, size_y)

        # Define empty zones (1.5m at each end)
        empty_zone = 3.0
        obstacle_area_start = center_x - terrain_size[0] / 2 + empty_zone
        obstacle_area_end = center_x + terrain_size[0] / 2 - empty_zone

        # Place obstacles only in the middle area
        grid_size_x = 4
        grid_size_y = 3
        obst_count = grid_size_x * grid_size_y
        obstacle_area_width = obstacle_area_end - obstacle_area_start
        cell_size_x = obstacle_area_width / grid_size_x
        cell_size_y = terrain_size[1] / grid_size_y

        obst_centers = np.zeros((obst_count, 2))
        obst_centers[:, 0] = np.arange(obst_count) % grid_size_x * cell_size_x + cell_size_x / 2.0 + obstacle_area_start
        obst_centers[:, 1] = (
            np.arange(obst_count) // grid_size_x * cell_size_y + cell_size_y / 2.0 + center_y - terrain_size[1] / 2
        )
        # obst_centers[:, 0] += np.random.uniform(-0.5, 0.5, obst_count)
        # obst_centers[:, 1] += np.random.uniform(-0.5, 0.5, obst_count)
        obst_rotations = np.random.uniform(0, 2 * np.pi, obst_count)

        # staggered offsets for each row
        for i in range(obst_count):
            row = i % grid_size_x
            offset = (row % 2) * (cell_size_y / 4.0) + ((row + 1) % 2) * (-cell_size_y / 4.0)
            obst_centers[i, 0] += np.random.uniform(-0.1, 0.1)
            obst_centers[i, 1] += np.random.uniform(-0.1, 0.1) + offset

        # Add random composite obstacles in the middle area only
        for i in range(obst_count):
            # self.create_simple_composite_obstacle(
            #     obst_centers[i, 0], obst_centers[i, 1], obst_rotations[i], meshes_list, box_size_range=(0.1, 0.2)
            # )
            self.create_jagged_wall(
                meshes_list,
                (obst_centers[i, 0], obst_centers[i, 1] - 0.6),
                (obst_centers[i, 0], obst_centers[i, 1] + 0.6),
                (0.1, 0.2),
            )

    def create_env_scatter_static_sparse(self, meshes_list, center_x, center_y, size_x, size_y):
        terrain_size = (size_x, size_y)

        # Create enclosing walls
        self.create_enclosing_walls(meshes_list, center_x, center_y, size_x, size_y)

        # Define empty zones (1.5m at each end)
        empty_zone = 3.0
        obstacle_area_start = center_x - terrain_size[0] / 2 + empty_zone
        obstacle_area_end = center_x + terrain_size[0] / 2 - empty_zone

        # Place obstacles only in the middle area
        grid_size_x = 3
        grid_size_y = 3
        obst_count = grid_size_x * grid_size_y
        obstacle_area_width = obstacle_area_end - obstacle_area_start
        cell_size_x = obstacle_area_width / grid_size_x
        cell_size_y = terrain_size[1] / grid_size_y

        obst_centers = np.zeros((obst_count, 2))
        obst_centers[:, 0] = np.arange(obst_count) % grid_size_x * cell_size_x + cell_size_x / 2.0 + obstacle_area_start
        obst_centers[:, 1] = (
            np.arange(obst_count) // grid_size_x * cell_size_y + cell_size_y / 2.0 + center_y - terrain_size[1] / 2
        )
        # obst_centers[:, 0] += np.random.uniform(-0.5, 0.5, obst_count)
        # obst_centers[:, 1] += np.random.uniform(-0.5, 0.5, obst_count)
        obst_rotations = np.random.uniform(0, 2 * np.pi, obst_count)

        # staggered offsets for each row
        for i in range(obst_count):
            row = i % grid_size_x
            offset = (row % 2) * (cell_size_y / 4.0) + ((row + 1) % 2) * (-cell_size_y / 4.0)
            obst_centers[i, 0] += np.random.uniform(-0.1, 0.1)
            obst_centers[i, 1] += np.random.uniform(-0.1, 0.1) + offset

        # Add random composite obstacles in the middle area only
        for i in range(obst_count):
            self.create_simple_composite_obstacle(
                obst_centers[i, 0], obst_centers[i, 1], obst_rotations[i], meshes_list, box_size_range=(0.1, 0.3)
            )
            # self.create_jagged_wall(
            #     meshes_list,
            #     (obst_centers[i, 0], obst_centers[i, 1] - 0.1),
            #     (obst_centers[i, 0], obst_centers[i, 1] + 0.1),
            #     (0.1, 0.2),
            # )

    def create_env_scatter_static_dense(self, meshes_list, center_x, center_y, size_x, size_y):
        terrain_size = (size_x, size_y)

        # Create enclosing walls
        self.create_enclosing_walls(meshes_list, center_x, center_y, size_x, size_y)

        # Define empty zones (1.5m at each end)
        empty_zone = 3.0
        obstacle_area_start = center_x - terrain_size[0] / 2 + empty_zone
        obstacle_area_end = center_x + terrain_size[0] / 2 - empty_zone

        # Place obstacles only in the middle area
        grid_size_x = 4
        grid_size_y = 4
        obst_count = grid_size_x * grid_size_y
        obstacle_area_width = obstacle_area_end - obstacle_area_start
        cell_size_x = obstacle_area_width / grid_size_x
        cell_size_y = terrain_size[1] / grid_size_y

        obst_centers = np.zeros((obst_count, 2))
        obst_centers[:, 0] = np.arange(obst_count) % grid_size_x * cell_size_x + cell_size_x / 2.0 + obstacle_area_start
        obst_centers[:, 1] = (
            np.arange(obst_count) // grid_size_x * cell_size_y + cell_size_y / 2.0 + center_y - terrain_size[1] / 2
        )
        # obst_centers[:, 0] += np.random.uniform(-0.5, 0.5, obst_count)
        # obst_centers[:, 1] += np.random.uniform(-0.5, 0.5, obst_count)
        obst_rotations = np.random.uniform(0, 2 * np.pi, obst_count)

        # staggered offsets for each row
        for i in range(obst_count):
            row = i % grid_size_x
            offset = (row % 2) * (cell_size_y / 4.0) + ((row + 1) % 2) * (-cell_size_y / 4.0)
            obst_centers[i, 0] += np.random.uniform(-0.1, 0.1)
            obst_centers[i, 1] += np.random.uniform(-0.1, 0.1) + offset

        # Add random composite obstacles in the middle area only
        for i in range(obst_count):
            self.create_simple_composite_obstacle(
                obst_centers[i, 0], obst_centers[i, 1], obst_rotations[i], meshes_list, box_size_range=(0.1, 0.3)
            )
            # self.create_jagged_wall(
            #     meshes_list,
            #     (obst_centers[i, 0], obst_centers[i, 1] - 0.1),
            #     (obst_centers[i, 0], obst_centers[i, 1] + 0.1),
            #     (0.1, 0.2),
            # )

    def create_env_scatter_dynamic(self, meshes_list, center_x, center_y, size_x, size_y):
        terrain_size = (size_x, size_y)

        # Create enclosing walls
        self.create_enclosing_walls(meshes_list, center_x, center_y, size_x, size_y)

        # Define empty zones (1.5m at each end)
        empty_zone = 3.0
        obstacle_area_start = center_x - terrain_size[0] / 2 + empty_zone
        obstacle_area_end = center_x + terrain_size[0] / 2 - empty_zone

        # Place obstacles only in the middle area
        grid_size_x = 4
        grid_size_y = 4
        obst_count = grid_size_x * grid_size_y
        obstacle_area_width = obstacle_area_end - obstacle_area_start
        cell_size_x = obstacle_area_width / grid_size_x
        cell_size_y = terrain_size[1] / grid_size_y

        obst_centers = np.zeros((obst_count, 2))
        obst_centers[:, 0] = np.arange(obst_count) % grid_size_x * cell_size_x + cell_size_x / 2.0 + obstacle_area_start
        obst_centers[:, 1] = (
            np.arange(obst_count) // grid_size_x * cell_size_y + cell_size_y / 2.0 + center_y - terrain_size[1] / 2
        )
        # obst_centers[:, 0] += np.random.uniform(-0.5, 0.5, obst_count)
        # obst_centers[:, 1] += np.random.uniform(-0.5, 0.5, obst_count)
        obst_rotations = np.random.uniform(0, 2 * np.pi, obst_count)

        # staggered offsets for each row
        for i in range(obst_count):
            row = i % grid_size_x
            offset = (row % 2) * (cell_size_y / 4.0) + ((row + 1) % 2) * (-cell_size_y / 4.0)
            obst_centers[i, 0] += np.random.uniform(-0.1, 0.1)
            obst_centers[i, 1] += np.random.uniform(-0.1, 0.1) + offset

        # Add random composite obstacles in the middle area only
        for i in range(obst_count):
            self.create_simple_composite_obstacle(
                obst_centers[i, 0], obst_centers[i, 1], obst_rotations[i], meshes_list
            )

    def create_env_scatter_dynamic_sparse(self, meshes_list, center_x, center_y, size_x, size_y):
        terrain_size = (size_x, size_y)

        # Create enclosing walls
        self.create_enclosing_walls(meshes_list, center_x, center_y, size_x, size_y)

        # Define empty zones (1.5m at each end)
        empty_zone = 3.0
        obstacle_area_start = center_x - terrain_size[0] / 2 + empty_zone
        obstacle_area_end = center_x + terrain_size[0] / 2 - empty_zone

        # Place obstacles only in the middle area
        grid_size_x = 3
        grid_size_y = 3
        obst_count = grid_size_x * grid_size_y
        obstacle_area_width = obstacle_area_end - obstacle_area_start
        cell_size_x = obstacle_area_width / grid_size_x
        cell_size_y = terrain_size[1] / grid_size_y

        obst_centers = np.zeros((obst_count, 2))
        obst_centers[:, 0] = np.arange(obst_count) % grid_size_x * cell_size_x + cell_size_x / 2.0 + obstacle_area_start
        obst_centers[:, 1] = (
            np.arange(obst_count) // grid_size_x * cell_size_y + cell_size_y / 2.0 + center_y - terrain_size[1] / 2
        )
        # obst_centers[:, 0] += np.random.uniform(-0.5, 0.5, obst_count)
        # obst_centers[:, 1] += np.random.uniform(-0.5, 0.5, obst_count)
        obst_rotations = np.random.uniform(0, 2 * np.pi, obst_count)

        # staggered offsets for each row
        for i in range(obst_count):
            row = i % grid_size_x
            offset = (row % 2) * (cell_size_y / 4.0) + ((row + 1) % 2) * (-cell_size_y / 4.0)
            obst_centers[i, 0] += np.random.uniform(-0.1, 0.1)
            obst_centers[i, 1] += np.random.uniform(-0.1, 0.1) + offset

        # Add random composite obstacles in the middle area only
        for i in range(obst_count):
            self.create_simple_composite_obstacle(
                obst_centers[i, 0], obst_centers[i, 1], obst_rotations[i], meshes_list, box_size_range=(0.1, 0.3)
            )

    def create_env_return(self, meshes_list, center_x, center_y, size_x, size_y):
        terrain_size = (size_x, size_y)

        # Create enclosing walls
        self.create_enclosing_walls(meshes_list, center_x, center_y, size_x, size_y)

        # Define empty zones (1.5m at each end)
        empty_zone = 3.0
        obstacle_area_start = center_x - terrain_size[0] / 2 + empty_zone
        obstacle_area_end = center_x + terrain_size[0] / 2 - empty_zone

        # Place obstacles only in the middle area
        grid_size_x = 3
        grid_size_y = 3
        obst_count = grid_size_x * grid_size_y
        obstacle_area_width = obstacle_area_end - obstacle_area_start
        cell_size_x = obstacle_area_width / grid_size_x
        cell_size_y = terrain_size[1] / grid_size_y

        obst_centers = np.zeros((obst_count, 2))
        obst_centers[:, 0] = np.arange(obst_count) % grid_size_x * cell_size_x + cell_size_x / 2.0 + obstacle_area_start
        obst_centers[:, 1] = (
            np.arange(obst_count) // grid_size_x * cell_size_y + cell_size_y / 2.0 + center_y - terrain_size[1] / 2
        )
        obst_centers[:, 0] += np.random.uniform(-1.0, 1.0, obst_count)
        obst_centers[:, 1] += np.random.uniform(-1.0, 1.0, obst_count)
        obst_rotations = np.random.uniform(0, 2 * np.pi, obst_count)

        # Add random composite obstacles in the middle area only
        for i in range(obst_count):
            dist_to_center = np.linalg.norm(obst_centers[i, :] - np.array([center_x, center_y]))
            if dist_to_center < 3.0:
                continue
            self.create_composite_obstacle(obst_centers[i, 0], obst_centers[i, 1], obst_rotations[i], meshes_list)

    def create_env_squeeze_2(self, meshes_list, center_x, center_y, size_x, size_y):
        terrain_size = (size_x, size_y)

        # Create enclosing walls
        self.create_enclosing_walls(meshes_list, center_x, center_y, size_x, size_y)

        # Define empty zones (1.5m at each end)
        empty_zone = 3.0
        obstacle_area_start = center_x - terrain_size[0] / 2 + empty_zone
        obstacle_area_end = center_x + terrain_size[0] / 2 - empty_zone

        # Place obstacles only in the middle area
        grid_size_x = 3
        grid_size_y = 3
        obst_count = grid_size_x * grid_size_y
        obstacle_area_width = obstacle_area_end - obstacle_area_start
        cell_size_x = obstacle_area_width / grid_size_x
        cell_size_y = terrain_size[1] / grid_size_y

        obst_centers = np.zeros((obst_count, 2))
        obst_centers[:, 0] = np.arange(obst_count) % grid_size_x * cell_size_x + cell_size_x / 2.0 + obstacle_area_start
        obst_centers[:, 1] = (
            np.arange(obst_count) // grid_size_x * cell_size_y + cell_size_y / 2.0 + center_y - terrain_size[1] / 2
        )
        obst_centers[:, 0] += np.random.uniform(-1.0, 1.0, obst_count)
        obst_centers[:, 1] += np.random.uniform(-1.0, 1.0, obst_count)
        obst_rotations = np.random.uniform(0, 2 * np.pi, obst_count)

        # Add random composite obstacles in the middle area only
        for i in range(obst_count):
            dist_to_center = np.linalg.norm(obst_centers[i, :] - np.array([center_x, center_y]))
            if dist_to_center < 3.0:
                continue
            self.create_composite_obstacle(obst_centers[i, 0], obst_centers[i, 1], obst_rotations[i], meshes_list)

    def create_env_squeeze_3(self, meshes_list, center_x, center_y, size_x, size_y):
        terrain_size = (size_x, size_y)

        # Create enclosing walls
        self.create_enclosing_walls(meshes_list, center_x, center_y, size_x, size_y)

        # Define empty zones (1.5m at each end)
        empty_zone = 3.0
        obstacle_area_start = center_x - terrain_size[0] / 2 + empty_zone
        obstacle_area_end = center_x + terrain_size[0] / 2 - empty_zone

        # Place obstacles only in the middle area
        grid_size_x = 3
        grid_size_y = 3
        obst_count = grid_size_x * grid_size_y
        obstacle_area_width = obstacle_area_end - obstacle_area_start
        cell_size_x = obstacle_area_width / grid_size_x
        cell_size_y = terrain_size[1] / grid_size_y

        obst_centers = np.zeros((obst_count, 2))
        obst_centers[:, 0] = np.arange(obst_count) % grid_size_x * cell_size_x + cell_size_x / 2.0 + obstacle_area_start
        obst_centers[:, 1] = (
            np.arange(obst_count) // grid_size_x * cell_size_y + cell_size_y / 2.0 + center_y - terrain_size[1] / 2
        )
        obst_centers[:, 0] += np.random.uniform(-1.0, 1.0, obst_count)
        obst_centers[:, 1] += np.random.uniform(-1.0, 1.0, obst_count)
        obst_rotations = np.random.uniform(0, 2 * np.pi, obst_count)

        # Add random composite obstacles in the middle area only
        for i in range(obst_count):
            dist_to_center = np.linalg.norm(obst_centers[i, :] - np.array([center_x, center_y]))
            if dist_to_center < 3.0:
                continue
            self.create_composite_obstacle(obst_centers[i, 0], obst_centers[i, 1], obst_rotations[i], meshes_list)

    def create_env_maze(self, meshes_list, center_x, center_y, size_x, size_y):
        terrain_size = (size_x, size_y)

        # Create enclosing walls
        self.create_enclosing_walls(meshes_list, center_x, center_y, size_x, size_y)

        cell_size = 2.0
        num_cells_x = int(terrain_size[0] / cell_size)
        num_cells_y = int(terrain_size[1] / cell_size)

        # Calculate start and goal positions (centered on Y axis)
        start_x = center_x - terrain_size[0] / 2 + 2.0
        start_y = center_y  # Centered on Y axis
        goal_x = center_x + terrain_size[0] / 2 - 2.0
        goal_y = center_y  # Centered on Y axis

        # Calculate which cell contains the start position
        start_cell_x = int((start_x - (center_x - terrain_size[0] / 2)) / cell_size)
        start_cell_y = int((start_y - (center_y - terrain_size[1] / 2)) / cell_size)

        # Generate maze using Prim's algorithm starting from the start cell
        maze = self._generate_maze_grid(
            num_cells_x,
            num_cells_y,
            start_cell=(start_cell_x, start_cell_y),
            loop_probability=0.15,
        )

        # Convert maze to walls
        # maze[i][j] has 4 bits: [top, right, bottom, left] where 1 means wall exists
        for i in range(num_cells_x):
            for j in range(num_cells_y):
                cell_center_x = center_x - terrain_size[0] / 2 + (i + 0.5) * cell_size
                cell_center_y = center_y - terrain_size[1] / 2 + (j + 0.5) * cell_size

                # Top wall
                if maze[i][j] & 0b1000:
                    start = (cell_center_x - cell_size / 2, cell_center_y + cell_size / 2)
                    end = (cell_center_x + cell_size / 2, cell_center_y + cell_size / 2)
                    self.create_jagged_wall(meshes_list, start, end)

                # Right wall
                if maze[i][j] & 0b0100:
                    start = (cell_center_x + cell_size / 2, cell_center_y - cell_size / 2)
                    end = (cell_center_x + cell_size / 2, cell_center_y + cell_size / 2)
                    self.create_jagged_wall(meshes_list, start, end)

                # Bottom wall
                if maze[i][j] & 0b0010:
                    start = (cell_center_x - cell_size / 2, cell_center_y - cell_size / 2)
                    end = (cell_center_x + cell_size / 2, cell_center_y - cell_size / 2)
                    self.create_jagged_wall(meshes_list, start, end)

                # Left wall
                if maze[i][j] & 0b0001:
                    start = (cell_center_x - cell_size / 2, cell_center_y - cell_size / 2)
                    end = (cell_center_x - cell_size / 2, cell_center_y + cell_size / 2)
                    self.create_jagged_wall(meshes_list, start, end)

        # Return the centered start and goal positions (aligned to cell centers)
        start_pos_x = center_x - terrain_size[0] / 2 + (start_cell_x + 0.5) * cell_size
        start_pos_y = center_y
        goal_cell_x = int((goal_x - (center_x - terrain_size[0] / 2)) / cell_size)
        goal_pos_x = center_x - terrain_size[0] / 2 + (goal_cell_x + 0.5) * cell_size
        goal_pos_y = center_y

        return np.array([start_pos_x, start_pos_y]), np.array([goal_pos_x, goal_pos_y])

    def create_env_dynamic_maze(self, meshes_list, center_x, center_y, size_x, size_y, maze_layout: str):
        # Create enclosing walls
        self.create_enclosing_walls(meshes_list, center_x, center_y, size_x, size_y)

        # Parse and create walls from the maze layout
        start_pos, goal_pos, obstacle_positions = self._create_maze_from_string(
            meshes_list, maze_layout, center_x, center_y, size_x, size_y
        )

        return start_pos, goal_pos, obstacle_positions

    def _create_maze_from_string(self, meshes_list, maze_string, center_x, center_y, size_x, size_y):
        """Parse a maze string definition and create walls.

        Args:
            meshes_list: List to append wall meshes to
            maze_string: Multi-line string defining the maze grid
            center_x: X coordinate of terrain center
            center_y: Y coordinate of terrain center
            size_x: Width of terrain
            size_y: Height of terrain

        Returns:
            Tuple of (start_position, goal_position, obstacle_positions) as numpy arrays
        """
        # Parse the maze string
        lines = [line.strip() for line in maze_string.strip().split("\n") if line.strip()]
        grid = [line.split() for line in lines]

        # Get grid dimensions
        num_rows = len(grid)
        num_cols = len(grid[0]) if grid else 0

        # Validate start, goal, and obstacle markers
        start_count = 0
        goal_count = 0
        start_pos = None
        goal_pos = None
        obstacle_positions = []

        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                cell_str = grid[row_idx][col_idx]
                if "s" in cell_str:
                    start_count += 1
                    start_pos = (row_idx, col_idx)
                if "g" in cell_str:
                    goal_count += 1
                    goal_pos = (row_idx, col_idx)
                if "o" in cell_str:
                    obstacle_positions.append((row_idx, col_idx))

        # Validate exactly one start and one goal
        if start_count != 1:
            raise ValueError(f"Maze must have exactly one start cell ('s'), found {start_count}")
        if goal_count != 1:
            raise ValueError(f"Maze must have exactly one goal cell ('g'), found {goal_count}")

        # Calculate cell sizes
        cell_size_x = size_x / num_cols
        cell_size_y = size_y / num_rows

        # Calculate origin (top-left corner of the grid)
        origin_x = center_x - size_x / 2
        origin_y = center_y - size_y / 2

        # Create walls for each cell
        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                cell_str = grid[row_idx][col_idx]

                # Calculate cell center position (flip X-axis)
                flipped_col_idx = num_cols - 1 - col_idx
                cell_center_x = origin_x + (flipped_col_idx + 0.5) * cell_size_x
                cell_center_y = origin_y + (row_idx + 0.5) * cell_size_y

                # Skip if no walls ('x' or cells that only contain markers)
                if cell_str == "x" or cell_str in ["s", "g", "o"]:
                    continue

                # Create walls based on direction characters
                # 'd' = down/bottom wall (at bottom edge of cell)
                if "d" in cell_str:
                    start = (cell_center_x - cell_size_x / 2, cell_center_y + cell_size_y / 2)
                    end = (cell_center_x + cell_size_x / 2, cell_center_y + cell_size_y / 2)
                    self.create_jagged_wall(meshes_list, start, end)

                # 'u' = up/top wall (at top edge of cell)
                if "u" in cell_str:
                    start = (cell_center_x - cell_size_x / 2, cell_center_y - cell_size_y / 2)
                    end = (cell_center_x + cell_size_x / 2, cell_center_y - cell_size_y / 2)
                    self.create_jagged_wall(meshes_list, start, end)

                # 'l' = left wall (becomes right wall after X-axis flip)
                if "l" in cell_str:
                    start = (cell_center_x + cell_size_x / 2, cell_center_y - cell_size_y / 2)
                    end = (cell_center_x + cell_size_x / 2, cell_center_y + cell_size_y / 2)
                    self.create_jagged_wall(meshes_list, start, end)

                # 'r' = right wall (becomes left wall after X-axis flip)
                if "r" in cell_str:
                    start = (cell_center_x - cell_size_x / 2, cell_center_y - cell_size_y / 2)
                    end = (cell_center_x - cell_size_x / 2, cell_center_y + cell_size_y / 2)
                    self.create_jagged_wall(meshes_list, start, end)

        # Calculate start and goal positions from marked cells (flip X-axis)
        start_row, start_col = start_pos
        goal_row, goal_col = goal_pos

        flipped_start_col = num_cols - 1 - start_col
        flipped_goal_col = num_cols - 1 - goal_col

        start_position = np.array(
            [origin_x + (flipped_start_col + 0.5) * cell_size_x, origin_y + (start_row + 0.5) * cell_size_y]
        )

        goal_position = np.array(
            [origin_x + (flipped_goal_col + 0.5) * cell_size_x, origin_y + (goal_row + 0.5) * cell_size_y]
        )

        # Calculate obstacle positions (flip X-axis)
        obstacle_world_positions = []
        for obs_row, obs_col in obstacle_positions:
            flipped_obs_col = num_cols - 1 - obs_col
            obs_position = np.array(
                [origin_x + (flipped_obs_col + 0.5) * cell_size_x, origin_y + (obs_row + 0.5) * cell_size_y]
            )
            obstacle_world_positions.append(obs_position)

        # Convert to numpy array
        if obstacle_world_positions:
            obstacle_array = np.array(obstacle_world_positions)
        else:
            obstacle_array = np.empty((0, 2))  # Empty array with correct shape

        return start_position, goal_position, obstacle_array

    def _generate_maze_grid(self, width, height, start_cell=None, loop_probability=0.1):
        """Generate a maze using Prim's algorithm with loop addition.

        Args:
            width: Number of cells in x direction
            height: Number of cells in y direction
            start_cell: Tuple (x, y) for the starting cell. If None, uses random cell.
            loop_probability: Probability of removing additional walls to create loops (0.0-1.0)

        Returns a 2D array where each cell contains a 4-bit value:
        bit 3 (0b1000): top wall
        bit 2 (0b0100): right wall
        bit 1 (0b0010): bottom wall
        bit 0 (0b0001): left wall
        """
        # Initialize maze with all walls
        maze = np.full((width, height), 0b1111, dtype=np.uint8)

        # Track cells in the maze
        in_maze = np.zeros((width, height), dtype=bool)

        # Wall list: each wall is (x1, y1, x2, y2, wall_bit1, wall_bit2)
        walls = []

        # Start from specified cell or random cell
        if start_cell is None:
            start = (np.random.randint(width), np.random.randint(height))
        else:
            start = start_cell

        in_maze[start] = True

        # Add walls of starting cell to wall list
        # Directions: (dx, dy, wall_bit_current, wall_bit_neighbor)
        directions = [
            (0, 1, 0b1000, 0b0010),  # top
            (1, 0, 0b0100, 0b0001),  # right
            (0, -1, 0b0010, 0b1000),  # bottom
            (-1, 0, 0b0001, 0b0100),  # left
        ]

        def add_walls(x, y):
            """Add all walls of a cell to the wall list."""
            for dx, dy, wall_current, wall_neighbor in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    walls.append((x, y, nx, ny, wall_current, wall_neighbor))

        add_walls(start[0], start[1])

        # Main Prim's algorithm loop
        while walls:
            # Pick a random wall
            wall_idx = np.random.randint(len(walls))
            x1, y1, x2, y2, wall1, wall2 = walls[wall_idx]
            walls.pop(wall_idx)

            # If the wall divides an "in" cell from an "out" cell
            if in_maze[x1, y1] and not in_maze[x2, y2]:
                # Remove the wall
                maze[x1, y1] &= ~wall1
                maze[x2, y2] &= ~wall2

                # Mark the new cell as in the maze
                in_maze[x2, y2] = True

                # Add walls of the newly added cell
                add_walls(x2, y2)

        # Add loops by randomly removing additional walls
        if loop_probability > 0:
            for x in range(width):
                for y in range(height):
                    for dx, dy, wall_current, wall_neighbor in directions:
                        nx, ny = x + dx, y + dy
                        # Only consider walls that still exist
                        if 0 <= nx < width and 0 <= ny < height and (maze[x, y] & wall_current):
                            # Remove wall with given probability
                            if np.random.random() < loop_probability:
                                maze[x, y] &= ~wall_current
                                maze[nx, ny] &= ~wall_neighbor

        return maze
