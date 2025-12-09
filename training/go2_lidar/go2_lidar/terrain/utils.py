import functools
import heapq
from collections import defaultdict

import numpy as np
import torch
import trimesh

TERRAIN_COUNTER = 0


class GridMapNavigation:
    """Navigation class for dead-end terrain with occupancy map and optimal path directions."""

    def __init__(
        self,
        occupancy_map,
        goal_positions_grid,
        cell_size,
        offset_x,
        offset_y,
        center_x,
        center_y,
        base_rotation,
        device="cuda",
    ):
        self.device = device
        self.occupancy_map = torch.tensor(
            occupancy_map, dtype=torch.bool, device=device
        )  # 2D boolean tensor: True = occupied
        self.cell_size = cell_size
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.center_x = center_x
        self.center_y = center_y
        self.base_rotation = base_rotation
        self.grid_rows, self.grid_cols = occupancy_map.shape

        # Store goal positions as grid indices
        self.goal_positions_grid = goal_positions_grid
        self.num_goals = len(goal_positions_grid)

        # 8-directional movement vectors (row_delta, col_delta)
        self.directions = np.array(
            [
                [-1, -1],  # Up-left
                [-1, 0],  # Up
                [-1, 1],  # Up-right
                [0, -1],  # Left
                [0, 1],  # Right
                [1, -1],  # Down-left
                [1, 0],  # Down
                [1, 1],  # Down-right
            ]
        )

        # Corresponding unit vectors for navigation
        # These should be (dy, dx) since row corresponds to y and col corresponds to x
        self.direction_vectors = np.array(
            [
                [-1, -1],  # Up-left: dy=-1, dx=-1
                [-1, 0],  # Up: dy=-1, dx=0
                [-1, 1],  # Up-right: dy=-1, dx=1
                [0, -1],  # Left: dy=0, dx=-1
                [0, 1],  # Right: dy=0, dx=1
                [1, -1],  # Down-left: dy=1, dx=-1
                [1, 0],  # Down: dy=1, dx=0
                [1, 1],  # Down-right: dy=1, dx=1
            ],
            dtype=float,
        )
        # Normalize diagonal directions
        for i in [0, 2, 5, 7]:  # Diagonal indices
            self.direction_vectors[i] = self.direction_vectors[i] / np.sqrt(2)

        # Compute optimal direction maps for each goal
        self.direction_maps = []
        for i, goal_grid in enumerate(goal_positions_grid):
            print(f"Computing optimal directions for goal {i + 1}/{self.num_goals} at grid {goal_grid}   \r", end="")
            direction_map = self._compute_optimal_directions(goal_grid)
            self.direction_maps.append(torch.tensor(direction_map, dtype=torch.float32, device=device))
        print()

        # Stack direction maps for efficient batched access
        self.direction_maps_tensor = torch.stack(self.direction_maps)  # Shape: (num_goals, grid_rows, grid_cols, 2)

        # Precompute rotation matrices
        cos_rot = np.cos(-self.base_rotation)
        sin_rot = np.sin(-self.base_rotation)
        self.cos_rot = torch.tensor(cos_rot, dtype=torch.float32, device=device)
        self.sin_rot = torch.tensor(sin_rot, dtype=torch.float32, device=device)

    def _compute_optimal_directions(self, goal_grid):
        """Compute optimal directions from all cells to a specific goal using Dijkstra."""
        goal_row, goal_col = goal_grid

        # Initialize distance and direction maps (keep as numpy for Dijkstra)
        distances = np.full((self.grid_rows, self.grid_cols), np.inf)
        direction_map = np.full((self.grid_rows, self.grid_cols, 2), np.nan)

        # Priority queue: (distance, row, col)
        pq = [(0, goal_row, goal_col)]
        distances[goal_row, goal_col] = 0
        direction_map[goal_row, goal_col] = [0, 0]  # Goal has no direction

        # Convert occupancy map to numpy for this computation
        occupancy_np = self.occupancy_map.cpu().numpy()

        while pq:
            current_dist, row, col = heapq.heappop(pq)

            if current_dist > distances[row, col]:
                continue

            # Check all 8 neighbors
            for dir_idx, (dr, dc) in enumerate(self.directions):
                new_row, new_col = row + dr, col + dc

                # Check bounds
                if not (0 <= new_row < self.grid_rows and 0 <= new_col < self.grid_cols):
                    continue

                # Check if cell is occupied
                if occupancy_np[new_row, new_col]:
                    continue

                # For diagonal moves, check if we can actually move diagonally
                # Block diagonal movement if either neighboring cell is occupied
                if abs(dr) + abs(dc) == 2:  # Diagonal move
                    # Check the two neighboring cells that share edges with both current and target
                    neighbor1_row, neighbor1_col = row + dr, col  # Horizontal neighbor
                    neighbor2_row, neighbor2_col = row, col + dc  # Vertical neighbor

                    # Block diagonal if either neighbor is occupied
                    if (
                        0 <= neighbor1_row < self.grid_rows
                        and 0 <= neighbor1_col < self.grid_cols
                        and occupancy_np[neighbor1_row, neighbor1_col]
                    ) or (
                        0 <= neighbor2_row < self.grid_rows
                        and 0 <= neighbor2_col < self.grid_cols
                        and occupancy_np[neighbor2_row, neighbor2_col]
                    ):
                        continue

                # Calculate new distance (diagonal moves cost sqrt(2))
                move_cost = np.sqrt(2) if abs(dr) + abs(dc) == 2 else 1.0
                new_dist = current_dist + move_cost

                if new_dist < distances[new_row, new_col]:
                    distances[new_row, new_col] = new_dist
                    # Store direction from neighbor toward current (toward goal)
                    # Since (dr, dc) moves from current to neighbor,
                    # we need (-dr, -dc) to move from neighbor to current
                    direction_map[new_row, new_col] = [-dr, -dc]
                    heapq.heappush(pq, (new_dist, new_row, new_col))

        return direction_map

    def get_optimal_direction(self, x, y, goal_index):
        """Convert global coordinates to optimal direction toward specified goal.

        Args:
            x: Tensor of shape (...,) containing x coordinates
            y: Tensor of shape (...,) containing y coordinates
            goal_index: Tensor of shape (...,) containing goal indices (0 to num_goals-1)

        Returns:
            Tensor of shape (..., 2): Unit vectors [dx, dy] pointing toward goals, or [0, 0] if invalid
        """
        # Ensure inputs are tensors on the right device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        if not isinstance(goal_index, torch.Tensor):
            goal_index = torch.tensor(goal_index, dtype=torch.long, device=self.device)

        # Move to device if needed
        x = x.to(self.device)
        y = y.to(self.device)
        goal_index = goal_index.to(self.device)

        # Get original shape and flatten for processing
        original_shape = x.shape
        x_flat = x.flatten()
        y_flat = y.flatten()
        goal_index_flat = goal_index.flatten()
        batch_size = x_flat.shape[0]

        # Convert global coordinates to local grid coordinates
        # First, translate to terrain-centered coordinates
        local_x = x_flat - self.center_x
        local_y = y_flat - self.center_y

        # Apply inverse rotation
        rotated_x = local_x * self.cos_rot - local_y * self.sin_rot
        rotated_y = local_x * self.sin_rot + local_y * self.cos_rot

        # Translate to grid coordinates
        grid_x = rotated_x - self.offset_x
        grid_y = rotated_y - self.offset_y

        # Convert to grid indices
        # Note: grid_x corresponds to column, grid_y corresponds to row
        col = (grid_x / self.cell_size).long()
        row = (grid_y / self.cell_size).long()

        # Check bounds
        valid_mask = (row >= 0) & (row < self.grid_rows) & (col >= 0) & (col < self.grid_cols)
        valid_goal_mask = (goal_index_flat >= 0) & (goal_index_flat < self.num_goals)
        valid_mask = valid_mask & valid_goal_mask

        # Initialize output
        directions = torch.zeros(batch_size, 2, dtype=torch.float32, device=self.device)

        # Only process valid indices
        if valid_mask.any():
            valid_indices = torch.where(valid_mask)[0]
            valid_row = row[valid_indices]
            valid_col = col[valid_indices]
            valid_goal_idx = goal_index_flat[valid_indices]

            # Get directions from the map using advanced indexing
            # direction_maps_tensor shape: (num_goals, grid_rows, grid_cols, 2)
            direction_values = self.direction_maps_tensor[valid_goal_idx, valid_row, valid_col]  # Shape: (num-valid, 2)

            # Check for NaN values
            nan_mask = torch.isnan(direction_values).any(dim=1)
            valid_direction_mask = ~nan_mask

            if valid_direction_mask.any():
                valid_direction_indices = valid_indices[valid_direction_mask]
                valid_directions = direction_values[valid_direction_mask]

                # The direction in the map is in grid coordinates (row, col) -> (dy, dx)
                # We need to convert this back to world coordinates
                dy = valid_directions[:, 0]
                dx = valid_directions[:, 1]

                # Rotate direction back to global frame
                global_dx = dx * self.cos_rot - dy * self.sin_rot
                global_dy = dx * self.sin_rot + dy * self.cos_rot

                directions[valid_direction_indices, 0] = global_dx
                directions[valid_direction_indices, 1] = global_dy

        # Reshape back to original shape
        return directions.view(*original_shape, 2)


class MazeGridNavigation:
    """Navigation class for maze-based terrain with edge-based wall representation."""

    def __init__(
        self,
        grid_rows,
        grid_cols,
        goal_positions_grid,
        cell_size,
        offset_x,
        offset_y,
        center_x,
        center_y,
        base_rotation,
        device="cuda",
    ):
        # Interpret inputs as width (x) then height (y)
        width_cols = grid_rows
        height_rows = grid_cols
        self.grid_rows = height_rows
        self.grid_cols = width_cols
        self.device = device
        self.cell_size = cell_size
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.center_x = center_x
        self.center_y = center_y
        self.base_rotation = base_rotation

        # Convert provided (x, y) goal indices to (row, col)
        self.goal_positions_grid = [(int(gy), int(gx)) for gx, gy in goal_positions_grid]
        self.num_goals = len(self.goal_positions_grid)

        # Initialize fully connected graph (4 directions, no diagonals)
        self.neighbors = {}
        for row in range(grid_rows):
            for col in range(grid_cols):
                cell = (row, col)
                self.neighbors[cell] = []
                # Up, Right, Down, Left
                for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < grid_rows and 0 <= new_col < grid_cols:
                        self.neighbors[cell].append((new_row, new_col))

        # 4-directional movement vectors (row_delta, col_delta)
        self.directions = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])  # Up, Right, Down, Left

        # Corresponding unit vectors for navigation (dy, dx)
        self.direction_vectors = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=float)

        # Direction maps and distance maps will be computed after walls are added
        self.direction_maps = None
        self.direction_maps_tensor = None
        self.distance_maps = None
        self.distance_maps_tensor = None

        # Precompute rotation matrices
        cos_rot = np.cos(-self.base_rotation)
        sin_rot = np.sin(-self.base_rotation)
        self.cos_rot = torch.tensor(cos_rot, dtype=torch.float32, device=device)
        self.sin_rot = torch.tensor(sin_rot, dtype=torch.float32, device=device)

    def add_walls(self, maze_array):
        maze_width, maze_height = maze_array.shape

        for mx in range(maze_width):
            for my in range(maze_height):
                cell = (my, mx)
                walls = maze_array[mx, my]

                # Remove edges bidirectionally to maintain graph symmetry
                # Top wall (0b1000) - blocks movement in +Y direction (row+1)
                if walls & 0b1000:
                    neighbor = (my + 1, mx)
                    if neighbor in self.neighbors[cell]:
                        self.neighbors[cell].remove(neighbor)
                    if 0 <= neighbor[0] < self.grid_rows and 0 <= neighbor[1] < self.grid_cols:
                        if cell in self.neighbors[neighbor]:
                            self.neighbors[neighbor].remove(cell)

                # Right wall (0b0100) - blocks movement in +X direction (col+1)
                if walls & 0b0100:
                    neighbor = (my, mx + 1)
                    if neighbor in self.neighbors[cell]:
                        self.neighbors[cell].remove(neighbor)
                    if 0 <= neighbor[0] < self.grid_rows and 0 <= neighbor[1] < self.grid_cols:
                        if cell in self.neighbors[neighbor]:
                            self.neighbors[neighbor].remove(cell)

                # Bottom wall (0b0010) - blocks movement in -Y direction (row-1)
                if walls & 0b0010:
                    neighbor = (my - 1, mx)
                    if neighbor in self.neighbors[cell]:
                        self.neighbors[cell].remove(neighbor)
                    if 0 <= neighbor[0] < self.grid_rows and 0 <= neighbor[1] < self.grid_cols:
                        if cell in self.neighbors[neighbor]:
                            self.neighbors[neighbor].remove(cell)

                # Left wall (0b0001) - blocks movement in -X direction (col-1)
                if walls & 0b0001:
                    neighbor = (my, mx - 1)
                    if neighbor in self.neighbors[cell]:
                        self.neighbors[cell].remove(neighbor)
                    if 0 <= neighbor[0] < self.grid_rows and 0 <= neighbor[1] < self.grid_cols:
                        if cell in self.neighbors[neighbor]:
                            self.neighbors[neighbor].remove(cell)

        # Compute optimal direction maps and distance maps after walls are added
        self.direction_maps = []
        self.distance_maps = []
        for i, goal_grid in enumerate(self.goal_positions_grid):
            print(f"Computing optimal directions for goal {i + 1}/{self.num_goals} at grid {goal_grid}   \r", end="")
            direction_map, distance_map = self._compute_optimal_directions(goal_grid)
            self.direction_maps.append(torch.tensor(direction_map, dtype=torch.float32, device=self.device))
            self.distance_maps.append(torch.tensor(distance_map, dtype=torch.float32, device=self.device))
        print()

        # Stack direction maps and distance maps for efficient batched access
        self.direction_maps_tensor = torch.stack(self.direction_maps)
        self.distance_maps_tensor = torch.stack(self.distance_maps)

    def _compute_optimal_directions(self, goal_grid):
        """Compute optimal directions and distances from all cells to a specific goal using Dijkstra.

        Returns:
            direction_map: numpy array of shape (grid_rows, grid_cols, 2) containing direction vectors
            distance_map: numpy array of shape (grid_rows, grid_cols) containing distances
        """
        # goal_grid already stored as (row, col)
        goal_row, goal_col = goal_grid

        # Initialize distance and direction maps
        distances = np.full((self.grid_rows, self.grid_cols), np.inf)
        direction_map = np.full((self.grid_rows, self.grid_cols, 2), np.nan)

        # Priority queue: (distance, row, col)
        pq = [(0, goal_row, goal_col)]
        distances[goal_row, goal_col] = 0
        direction_map[goal_row, goal_col] = [0, 0]  # Goal has no direction

        while pq:
            current_dist, row, col = heapq.heappop(pq)

            if current_dist > distances[row, col]:
                continue

            # Check all valid neighbors (edges not blocked by walls)
            for neighbor_row, neighbor_col in self.neighbors[(row, col)]:
                # Calculate movement direction FROM current TO neighbor
                dr = neighbor_row - row
                dc = neighbor_col - col

                # Calculate new distance (all moves cost 1.0)
                new_dist = current_dist + 1.0

                if new_dist < distances[neighbor_row, neighbor_col]:
                    distances[neighbor_row, neighbor_col] = new_dist
                    # Store NORMALIZED direction from neighbor toward current (toward goal)
                    # Since (dr, dc) moves from current to neighbor,
                    # we need (-dr, -dc) to move from neighbor to current
                    direction_vector = np.array([-dr, -dc], dtype=float)
                    # Normalize to unit vector
                    magnitude = np.linalg.norm(direction_vector)
                    if magnitude > 0:
                        direction_vector = direction_vector / magnitude
                    direction_map[neighbor_row, neighbor_col] = direction_vector
                    heapq.heappush(pq, (new_dist, neighbor_row, neighbor_col))

        return direction_map, distances

    def get_optimal_direction(self, x, y, goal_index):
        """Convert global coordinates to optimal direction toward specified goal.

        Args:
            x: Tensor of shape (...,) containing x coordinates
            y: Tensor of shape (...,) containing y coordinates
            goal_index: Tensor of shape (...,) containing goal indices (0 to num_goals-1)

        Returns:
            Tensor of shape (..., 2): Unit vectors [dx, dy] pointing toward goals, or [0, 0] if invalid
        """
        # Ensure inputs are tensors on the right device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        if not isinstance(goal_index, torch.Tensor):
            goal_index = torch.tensor(goal_index, dtype=torch.long, device=self.device)

        # Move to device if needed
        x = x.to(self.device)
        y = y.to(self.device)
        goal_index = goal_index.to(self.device)

        # Get original shape and flatten for processing
        original_shape = x.shape
        x_flat = x.flatten()
        y_flat = y.flatten()
        goal_index_flat = goal_index.flatten()
        batch_size = x_flat.shape[0]

        # Convert global coordinates to local grid coordinates
        # First, translate to terrain-centered coordinates
        local_x = x_flat - self.center_x
        local_y = y_flat - self.center_y

        # Apply inverse rotation
        rotated_x = local_x * self.cos_rot - local_y * self.sin_rot
        rotated_y = local_x * self.sin_rot + local_y * self.cos_rot

        # Translate to grid coordinates
        grid_x = rotated_x - self.offset_x
        grid_y = rotated_y - self.offset_y

        # Convert to grid indices
        col = (grid_x / self.cell_size).long()
        row = (grid_y / self.cell_size).long()

        # Check bounds
        valid_mask = (row >= 0) & (row < self.grid_rows) & (col >= 0) & (col < self.grid_cols)
        valid_goal_mask = (goal_index_flat >= 0) & (goal_index_flat < self.num_goals)
        valid_mask = valid_mask & valid_goal_mask

        # Initialize output
        directions = torch.zeros(batch_size, 2, dtype=torch.float32, device=self.device)

        # Only process valid indices
        if valid_mask.any():
            valid_indices = torch.where(valid_mask)[0]
            valid_row = row[valid_indices]
            valid_col = col[valid_indices]
            valid_goal_idx = goal_index_flat[valid_indices]

            # Get directions from the map
            direction_values = self.direction_maps_tensor[valid_goal_idx, valid_row, valid_col]

            # Check for NaN values
            nan_mask = torch.isnan(direction_values).any(dim=1)
            valid_direction_mask = ~nan_mask

            if valid_direction_mask.any():
                valid_direction_indices = valid_indices[valid_direction_mask]
                valid_directions = direction_values[valid_direction_mask]

                # The direction in the map is in grid coordinates (row, col) -> (dy, dx)
                dy = valid_directions[:, 0]
                dx = valid_directions[:, 1]

                # Rotate direction back to global frame
                global_dx = dx * self.cos_rot - dy * self.sin_rot
                global_dy = dx * self.sin_rot + dy * self.cos_rot

                directions[valid_direction_indices, 0] = global_dx
                directions[valid_direction_indices, 1] = global_dy

        # Reshape back to original shape
        return directions.view(*original_shape, 2)


def add_random_obstacles(func):
    @functools.wraps(func)
    def wrapper(difficulty, cfg):
        meshes_list, origin = func(difficulty, cfg)
        obst_count = int(difficulty * 7)
        obst_radius = 0.4
        obst_centers = np.zeros((obst_count, 2))
        border_width = 0
        if hasattr(cfg, "border_width"):
            border_width = cfg.border_width
        border_length = 2.0
        terrain_size = (cfg.size[0] - 2 * border_length, cfg.size[1] - 2 * border_width)
        obst_centers[:, 0] = np.random.uniform(0, terrain_size[0], obst_count) + border_length
        obst_centers[:, 1] = np.random.uniform(0, terrain_size[1], obst_count) + border_width
        close_to_center = np.linalg.norm(obst_centers - np.array([cfg.size[0] / 2, cfg.size[1] / 2]), axis=1) < 0.5
        obst_centers[close_to_center, :2] += 1.0
        for i in range(obst_count):
            cylinder = trimesh.creation.cylinder(radius=obst_radius, height=4.0)
            cylinder.apply_translation([obst_centers[i, 0], obst_centers[i, 1], 0.0])
            meshes_list.append(cylinder)
        return meshes_list, origin

    return wrapper


def _create_composite_obstacle(center_x, center_y, base_rotation, meshes_list, box_size_range=(0.1, 0.3)):
    """Creates a composite obstacle made of 3-4 smaller boxes."""
    num_boxes = np.random.randint(3, 5)  # 3 or 4 boxes
    pattern = np.random.choice(["row", "L_shape", "grid_2x2", "T_shape"])

    # Configurable parameters
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


def _create_composite_obstacle_nav(center_x, center_y, base_rotation, meshes_list, num_boxes=None):
    """Creates a composite obstacle made of smaller boxes.

    Args:
        center_x: X coordinate of obstacle center
        center_y: Y Coordinate of obstacle center
        base_rotation: Base rotation to apply
        meshes_list: List to append meshes to
        num_boxes: Number of boxes to create (3-4 if None)
    """
    if num_boxes is None:
        num_boxes = np.random.randint(3, 5)  # 3 or 4 boxes
    pattern = np.random.choice(["row", "L_shape", "grid", "T_shape"])

    # Configurable parameters
    base_size = np.random.uniform(0.1, 0.2)
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
        # L-shaped pattern - create an L with specified number of boxes
        # Split boxes between horizontal and vertical arms
        h_boxes = num_boxes // 2 + 1
        v_boxes = num_boxes - h_boxes + 1  # +1 because corner is shared

        positions = []
        # Horizontal arm
        for i in range(h_boxes):
            positions.append((i, 0))
        # Vertical arm (skip corner which is already added)
        for i in range(1, v_boxes):
            positions.append((0, i))

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

    elif pattern == "grid":
        # Grid pattern - arrange boxes in a roughly square grid
        grid_size = int(np.ceil(np.sqrt(num_boxes)))
        positions = []
        for i in range(num_boxes):
            dx = i % grid_size
            dy = i // grid_size
            positions.append((dx, dy))

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
        # T-shaped pattern - create a T with specified number of boxes
        # Top horizontal bar and vertical stem
        top_boxes = (num_boxes + 1) // 2
        stem_boxes = num_boxes - top_boxes + 1  # +1 because center is shared

        positions = []
        # Top horizontal bar centered at origin
        for i in range(top_boxes):
            dx = i - (top_boxes - 1) / 2
            positions.append((dx, 1))
        # Vertical stem (skip top which is already added)
        for i in range(stem_boxes - 1):
            positions.append((0, -i))

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
    for box, extents, local_pos in temp_boxes:
        # Apply composite rotation around origin
        box.apply_transform(trimesh.transformations.rotation_matrix(composite_rotation, [0, 0, 1]))
        # Translate to final position
        box.apply_translation([center_x, center_y, 0.0])
        meshes_list.append(box)


def _create_filter_obstacles(cfg, meshes_list):
    terrain_size = (cfg.size[0], cfg.size[1])
    grid_size = 5
    obst_count = grid_size**2
    cell_size = np.array(terrain_size) / grid_size
    obst_centers = np.zeros((obst_count, 2))
    obst_centers[:, 0] = np.arange(obst_count) % grid_size * cell_size[0] + cell_size[0] / 2.0
    obst_centers[:, 1] = np.arange(obst_count) // grid_size * cell_size[1] + cell_size[1] / 2.0
    obst_centers[:, 0] += np.random.uniform(-0.75, 0.75, obst_count)
    obst_centers[:, 1] += np.random.uniform(-0.75, 0.75, obst_count)
    obst_rotations = np.random.uniform(0, 2 * np.pi, obst_count)

    # add random composite obstacles
    for i in range(obst_count):
        _create_composite_obstacle(obst_centers[i, 0], obst_centers[i, 1], obst_rotations[i], meshes_list)


def _create_nav_obstacles(cfg, meshes_list):
    terrain_size = (cfg.size[0], cfg.size[1])
    grid_size = 5
    obst_count = grid_size**2
    cell_size = np.array(terrain_size) / grid_size
    obst_centers = np.zeros((obst_count, 2))
    obst_centers[:, 0] = np.arange(obst_count) % grid_size * cell_size[0] + cell_size[0] / 2.0
    obst_centers[:, 1] = np.arange(obst_count) // grid_size * cell_size[1] + cell_size[1] / 2.0
    obst_centers[:, 0] += np.random.uniform(-0.5, 0.5, obst_count)
    obst_centers[:, 1] += np.random.uniform(-0.5, 0.5, obst_count)
    obst_rotations = np.random.uniform(0, 2 * np.pi, obst_count)

    # add random composite obstacles
    for i in range(obst_count):
        _create_composite_obstacle(obst_centers[i, 0], obst_centers[i, 1], obst_rotations[i], meshes_list)


def _create_test_obstacles(cfg, meshes_list):
    terrain_size = (cfg.size[0], cfg.size[1])

    def create_jagged_wall(start_pos, end_pos):
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

    # Create enclosing walls
    # Left wall
    create_jagged_wall((0, 0), (0, terrain_size[1]))
    # Right wall
    create_jagged_wall((terrain_size[0], 0), (terrain_size[0], terrain_size[1]))
    # Bottom wall
    create_jagged_wall((0, 0), (terrain_size[0], 0))
    # Top wall
    create_jagged_wall((0, terrain_size[1]), (terrain_size[0], terrain_size[1]))

    # Define empty zones (1.5m at each end)
    empty_zone = 3.0
    obstacle_area_start = empty_zone
    obstacle_area_end = terrain_size[0] - empty_zone

    # Place obstacles only in the middle area
    grid_size_x = 6
    grid_size_y = 6
    obst_count = grid_size_x * grid_size_y
    obstacle_area_width = obstacle_area_end - obstacle_area_start
    cell_size_x = obstacle_area_width / grid_size_x
    cell_size_y = (terrain_size[1] - 0.0) / grid_size_y

    obst_centers = np.zeros((obst_count, 2))
    obst_centers[:, 0] = np.arange(obst_count) % grid_size_x * cell_size_x + cell_size_x / 2.0 + obstacle_area_start
    obst_centers[:, 1] = np.arange(obst_count) // grid_size_x * cell_size_y + cell_size_y / 2.0
    obst_centers[:, 0] += np.random.uniform(-1.0, 1.0, obst_count)
    obst_centers[:, 1] += np.random.uniform(-1.0, 1.0, obst_count)
    obst_rotations = np.random.uniform(0, 2 * np.pi, obst_count)

    # Add random composite obstacles in the middle area only
    for i in range(obst_count):
        _create_composite_obstacle(obst_centers[i, 0], obst_centers[i, 1], obst_rotations[i], meshes_list)


def _create_abs_obstacles(cfg, meshes_list):
    terrain_size = (cfg.size[0], cfg.size[1])
    grid_size = 5
    obst_count = grid_size**2
    cell_size = np.array(terrain_size) / grid_size
    obst_centers = np.zeros((obst_count, 2))
    obst_centers[:, 0] = np.arange(obst_count) % grid_size * cell_size[0] + cell_size[0] / 2.0
    obst_centers[:, 1] = np.arange(obst_count) // grid_size * cell_size[1] + cell_size[1] / 2.0
    obst_centers[:, 0] += np.random.uniform(-0.75, 0.75, obst_count)
    obst_centers[:, 1] += np.random.uniform(-0.75, 0.75, obst_count)
    obst_rotations = np.random.uniform(0, 2 * np.pi, obst_count)

    # add random composite obstacles
    for i in range(obst_count):
        _create_composite_obstacle(obst_centers[i, 0], obst_centers[i, 1], obst_rotations[i], meshes_list)


def add_walls(func):
    @functools.wraps(func)
    def wrapper(difficulty, cfg):
        meshes_list, origin = func(difficulty, cfg)

        # Seed random generator for deterministic walls per environment
        global TERRAIN_COUNTER
        np.random.seed(TERRAIN_COUNTER)
        TERRAIN_COUNTER += 1

        # get nominal difficulty
        difficulty = int(difficulty * 10) % 10

        if cfg.terrain_variant == "filter":
            _create_filter_obstacles(cfg, meshes_list)

        elif cfg.terrain_variant == "nav":
            # terrain mesh is generated in NavTerrainGenerator, so we do not need to add obstacles here.
            # _create_nav_obstacles(cfg, meshes_list)
            pass

        elif cfg.terrain_variant == "test":
            # terrain mesh is generated in TestTerrainGenerator
            # _create_test_obstacles(cfg, meshes_list)
            pass

        elif cfg.terrain_variant == "abs":
            _create_abs_obstacles(cfg, meshes_list)

        else:
            raise ValueError(f"Unknown terrain variant: {cfg.terrain_variant}")

        return meshes_list, origin

    return wrapper
