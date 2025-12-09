import os
import sys

import h5py
import numpy as np


class HDF5DatasetWriter_Ray:
    """
    A class to efficiently write large, sequential datasets to an HDF5 file.

    This is ideal for collecting data from a simulation where loading the entire
    dataset into RAM is not feasible. It writes data sample by sample,
    incrementally resizing the HDF5 datasets on disk.
    """

    def __init__(
        self,
        filename: str,
        num_total: int = 200000,
        history_frames: int = 10,
        grid_map_shape: tuple = (30, 180),
        gravity_dim: int = 3,
        ang_vel_dim: int = 3,
        gt_rays_dim: int = 180,
        overwrite: bool = False,
    ):
        """
        Initializes the HDF5 file and creates the datasets.

        Args:
            filename (str): The path to the HDF5 file to create.
            history_frames (int): The number of historical frames to store per sample (e.g., 10).
            grid_map_shape (tuple): The shape of a single grid map (e.g., (5, 180)).
            gravity_dim (int): The dimension of the gravity vector (e.g., 3).
            ang_vel_dim (int): The dimension of the angular velocity vector (e.g., 3).
            gt_rays_dim (int): The dimension of the ground truth ray vector (e.g., 180).
            overwrite (bool): If True, overwrite the file if it already exists.
        """
        if os.path.exists(filename) and not overwrite:
            raise ValueError(f"File {filename} already exists. Set overwrite=True to overwrite.")

        self.file = h5py.File(filename, "w")
        self.history_frames = history_frames
        self.grid_map_shape = grid_map_shape
        self.gravity_dim = gravity_dim
        self.ang_vel_dim = ang_vel_dim
        self.gt_rays_dim = gt_rays_dim
        self.idx = 0  # Internal counter for the number of samples
        self.num_total = num_total

        # --- Create the datasets ---
        # We create them with an initial size of 0 and make them resizable.
        # The 'maxshape=(None, ...)' allows the first dimension (the number of samples) to grow indefinitely.
        self.grid_maps_ds = self.file.create_dataset(
            "grid_maps",
            shape=(0, self.history_frames, *self.grid_map_shape),
            maxshape=(None, self.history_frames, *self.grid_map_shape),
            dtype="f4",  # 4-byte float (float32)
            compression="gzip",
            compression_opts=4,
        )
        self.gravity_vectors_ds = self.file.create_dataset(
            "gravity_vectors",
            shape=(0, self.history_frames, self.gravity_dim),
            maxshape=(None, self.history_frames, self.gravity_dim),
            dtype="f4",
            compression="gzip",
            compression_opts=4,
        )
        self.angular_velocity_ds = self.file.create_dataset(
            "angular_velocity",
            shape=(0, self.history_frames, self.ang_vel_dim),
            maxshape=(None, self.history_frames, self.ang_vel_dim),
            dtype="f4",
            compression="gzip",
            compression_opts=4,
        )
        self.ground_truth_rays_ds = self.file.create_dataset(
            "ground_truth_rays",
            shape=(0, self.gt_rays_dim),
            maxshape=(None, self.gt_rays_dim),
            dtype="f4",
            compression="gzip",
            compression_opts=4,
        )

        print(f"HDF5 file '{filename}' created successfully.")

    def add_data(
        self,
        num_envs: int,
        grid_maps: np.ndarray,
        gravity_vectors: np.ndarray,
        angular_velocity: np.ndarray,
        ground_truth_rays: np.ndarray,
    ):
        """
        Adds a single new data sample to the datasets.

        Args:
            grid_maps (np.ndarray): The sequence of grid maps for this sample.
                                    Shape: (history_frames, grid_map_height, grid_map_width)
            gravity_vectors (np.ndarray): The sequence of gravity vectors.
                                          Shape: (history_frames, 3)
            angular_velocity (np.ndarray): The sequence of angular velocity vectors.
                                          Shape: (history_frames, 3)
            ground_truth_rays (np.ndarray): The ground truth rays for the latest frame.
                                             Shape: (180,)
        """
        if self.idx >= self.num_total:
            return

        # --- Validate input shapes ---
        expected_gm_shape = (num_envs, self.history_frames, *self.grid_map_shape)
        if grid_maps.shape != expected_gm_shape:
            raise ValueError(f"Invalid grid_maps shape. Expected {expected_gm_shape}, got {grid_maps.shape}")

        expected_gv_shape = (num_envs, self.history_frames, self.gravity_dim)
        if gravity_vectors.shape != expected_gv_shape:
            raise ValueError(
                f"Invalid gravity_vectors shape. Expected {expected_gv_shape}, got {gravity_vectors.shape}"
            )

        expected_ang_vel_shape = (num_envs, self.history_frames, self.ang_vel_dim)
        if angular_velocity.shape != expected_ang_vel_shape:
            raise ValueError(
                f"Invalid angular_velocity shape. Expected {expected_ang_vel_shape}, got {angular_velocity.shape}"
            )

        if ground_truth_rays.shape != (num_envs, self.gt_rays_dim):
            raise ValueError(
                f"Invalid ground_truth_rays shape. Expected {(self.gt_rays_dim,)}, got {ground_truth_rays.shape}"
            )

        # --- Resize the datasets to accommodate the new sample ---
        self.grid_maps_ds.resize(self.idx + num_envs, axis=0)
        self.gravity_vectors_ds.resize(self.idx + num_envs, axis=0)
        self.angular_velocity_ds.resize(self.idx + num_envs, axis=0)
        self.ground_truth_rays_ds.resize(self.idx + num_envs, axis=0)

        # --- Write the new data ---
        self.grid_maps_ds[self.idx : self.idx + num_envs] = grid_maps
        self.gravity_vectors_ds[self.idx : self.idx + num_envs] = gravity_vectors
        self.angular_velocity_ds[self.idx : self.idx + num_envs] = angular_velocity
        self.ground_truth_rays_ds[self.idx : self.idx + num_envs] = ground_truth_rays

        # Increment the counter
        self.idx += num_envs

        print(f"{self.idx} samples written to HDF5 file.")

        if self.idx >= self.num_total:
            print(f"Reached the maximum number of samples ({self.num_total}). No more data will be added.")
            try:
                self.close()
                sys.exit(0)
            finally:
                print("HDF5 file closed after reaching the maximum number of samples. Exiting...")

    def close(self):
        """Closes the HDF5 file, ensuring all data is written to disk."""
        if self.file:
            self.file.close()
            self.file = None
            print(f"HDF5 file closed. Total samples written: {self.idx}")


class HDF5DatasetWriter_General:
    def __init__(
        self,
        filename: str,
        input_dim: int,
        output_dim: int,
        num_total: int = 200000,
        history_frames: int = 20,
        overwrite: bool = False,
    ):
        if os.path.exists(filename) and not overwrite:
            raise ValueError(f"File {filename} already exists. Set overwrite=True to overwrite.")

        self.file = h5py.File(filename, "w")
        self.history_frames = history_frames
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.idx = 0  # Internal counter for the number of samples
        self.num_total = num_total

        self.input_ds = self.file.create_dataset(
            "input",
            shape=(0, self.history_frames, self.input_dim),
            maxshape=(None, self.history_frames, self.input_dim),
            dtype="f4",  # 4-byte float (float32)
            compression="gzip",
            compression_opts=4,
        )
        self.output_ds = self.file.create_dataset(
            "output",
            shape=(0, self.output_dim),
            maxshape=(None, self.output_dim),
            dtype="f4",
            compression="gzip",
            compression_opts=4,
        )

        print(f"HDF5 file '{filename}' created successfully.")

    def add_data(self, num_envs: int, input: np.ndarray, output: np.ndarray):
        if self.idx >= self.num_total:
            return

        # --- Validate input shapes ---
        expected_input_shape = (num_envs, self.history_frames, self.input_dim)
        if input.shape != expected_input_shape:
            raise ValueError(f"Invalid input shape. Expected {expected_input_shape}, got {input.shape}")

        expected_output_shape = (num_envs, self.output_dim)
        if output.shape != expected_output_shape:
            raise ValueError(f"Invalid output shape. Expected {expected_output_shape}, got {output.shape}")

        # --- Resize the datasets to accommodate the new sample ---
        self.input_ds.resize(self.idx + num_envs, axis=0)
        self.output_ds.resize(self.idx + num_envs, axis=0)

        # --- Write the new data ---
        self.input_ds[self.idx : self.idx + num_envs] = input
        self.output_ds[self.idx : self.idx + num_envs] = output

        # Increment the counter
        self.idx += num_envs

        print(f"{self.idx} samples written to HDF5 file.")

        if self.idx >= self.num_total:
            print(f"Reached the maximum number of samples ({self.num_total}). No more data will be added.")
            try:
                self.close()
                sys.exit(0)
            finally:
                print("HDF5 file closed after reaching the maximum number of samples. Exiting...")

    def close(self):
        """Closes the HDF5 file, ensuring all data is written to disk."""
        if self.file:
            self.file.close()
            self.file = None
            print(f"HDF5 file closed. Total samples written: {self.idx}")


class HDF5DatasetWriter_RaySequential:
    """
    A class to efficiently write sequential frame data to an HDF5 file with episode tracking.

    This writer stores each episode as a contiguous block of frames, making episodes
    easy to load and navigate. Episode boundaries track the start and end indices of
    each episode in the global frame array.

    Each environment accumulates frames in a buffer. When an episode ends, the entire
    episode is written as a contiguous block to the HDF5 file.
    """

    def __init__(
        self,
        filename: str,
        num_envs: int,
        num_total_frames: int = 200000,
        grid_map_shape: tuple = (30, 180),
        gravity_dim: int = 3,
        ang_vel_dim: int = 3,
        gt_rays_dim: int = 180,
        overwrite: bool = False,
    ):
        """
        Initializes the HDF5 file and creates the datasets for sequential storage.

        Args:
            filename (str): The path to the HDF5 file to create.
            num_envs (int): Number of parallel environments.
            num_total_frames (int): Maximum total number of frames to store.
            grid_map_shape (tuple): The shape of a single grid map (e.g., (30, 180)).
            gravity_dim (int): The dimension of the gravity vector (e.g., 3).
            ang_vel_dim (int): The dimension of the angular velocity vector (e.g., 3).
            gt_rays_dim (int): The dimension of the ground truth ray vector (e.g., 180).
            overwrite (bool): If True, overwrite the file if it already exists.
        """
        if os.path.exists(filename) and not overwrite:
            raise ValueError(f"File {filename} already exists. Set overwrite=True to overwrite.")

        self.file = h5py.File(filename, "w")
        self.num_envs = num_envs
        self.grid_map_shape = grid_map_shape
        self.gravity_dim = gravity_dim
        self.ang_vel_dim = ang_vel_dim
        self.gt_rays_dim = gt_rays_dim
        self.frame_idx = 0  # Global frame counter (total frames written to file)
        self.num_total_frames = num_total_frames
        self.capacity_reached = False  # Flag to track if we've hit the limit

        # Episode buffers: accumulate frames for each environment until episode ends
        self.env_buffers = {
            "grid_maps": [[] for _ in range(num_envs)],
            "gravity_vectors": [[] for _ in range(num_envs)],
            "angular_velocity": [[] for _ in range(num_envs)],
            "ground_truth_rays": [[] for _ in range(num_envs)],
        }

        self.episode_boundaries = []  # List of (start_idx, end_idx) tuples
        self.timestep_count = 0  # Track number of timesteps for logging

        # Create datasets for frame data (stored sequentially)
        self.grid_maps_ds = self.file.create_dataset(
            "grid_maps",
            shape=(0, *self.grid_map_shape),
            maxshape=(None, *self.grid_map_shape),
            dtype="f4",
            compression="gzip",
            compression_opts=4,
        )
        self.gravity_vectors_ds = self.file.create_dataset(
            "gravity_vectors",
            shape=(0, self.gravity_dim),
            maxshape=(None, self.gravity_dim),
            dtype="f4",
            compression="gzip",
            compression_opts=4,
        )
        self.angular_velocity_ds = self.file.create_dataset(
            "angular_velocity",
            shape=(0, self.ang_vel_dim),
            maxshape=(None, self.ang_vel_dim),
            dtype="f4",
            compression="gzip",
            compression_opts=4,
        )
        self.ground_truth_rays_ds = self.file.create_dataset(
            "ground_truth_rays",
            shape=(0, self.gt_rays_dim),
            maxshape=(None, self.gt_rays_dim),
            dtype="f4",
            compression="gzip",
            compression_opts=4,
        )

        # Create dataset for episode boundaries (will be written at the end)
        self.episode_boundaries_ds = self.file.create_dataset(
            "episode_boundaries",
            shape=(0, 2),
            maxshape=(None, 2),
            dtype="i8",  # int64 for frame indices
        )

        # Store metadata
        self.file.attrs["num_envs"] = num_envs
        self.file.attrs["grid_map_shape"] = grid_map_shape
        self.file.attrs["gravity_dim"] = gravity_dim
        self.file.attrs["ang_vel_dim"] = ang_vel_dim
        self.file.attrs["gt_rays_dim"] = gt_rays_dim

        print(f"HDF5 sequential file '{filename}' created successfully.")
        print(f"Tracking {num_envs} parallel environments.")

    def add_frames(
        self,
        grid_maps: np.ndarray,
        gravity_vectors: np.ndarray,
        angular_velocity: np.ndarray,
        ground_truth_rays: np.ndarray,
    ):
        """
        Adds frames from all environments at the current timestep to their respective buffers.

        Args:
            grid_maps (np.ndarray): Grid maps from all envs. Shape: (num_envs, grid_map_height, grid_map_width)
            gravity_vectors (np.ndarray): Gravity vectors from all envs. Shape: (num_envs, 3)
            angular_velocity (np.ndarray): Angular velocity from all envs. Shape: (num_envs, 3)
            ground_truth_rays (np.ndarray): Ground truth rays from all envs. Shape: (num_envs, 180)
        """
        # Stop accepting frames if we've reached capacity
        if self.capacity_reached:
            return

        # Validate input shapes
        expected_gm_shape = (self.num_envs, *self.grid_map_shape)
        if grid_maps.shape != expected_gm_shape:
            raise ValueError(f"Invalid grid_maps shape. Expected {expected_gm_shape}, got {grid_maps.shape}")

        expected_gv_shape = (self.num_envs, self.gravity_dim)
        if gravity_vectors.shape != expected_gv_shape:
            raise ValueError(
                f"Invalid gravity_vectors shape. Expected {expected_gv_shape}, got {gravity_vectors.shape}"
            )

        expected_ang_vel_shape = (self.num_envs, self.ang_vel_dim)
        if angular_velocity.shape != expected_ang_vel_shape:
            raise ValueError(
                f"Invalid angular_velocity shape. Expected {expected_ang_vel_shape}, got {angular_velocity.shape}"
            )

        expected_gt_shape = (self.num_envs, self.gt_rays_dim)
        if ground_truth_rays.shape != expected_gt_shape:
            raise ValueError(
                f"Invalid ground_truth_rays shape. Expected {expected_gt_shape}, got {ground_truth_rays.shape}"
            )

        # Add frames to each environment's buffer
        for env_id in range(self.num_envs):
            self.env_buffers["grid_maps"][env_id].append(grid_maps[env_id])
            self.env_buffers["gravity_vectors"][env_id].append(gravity_vectors[env_id])
            self.env_buffers["angular_velocity"][env_id].append(angular_velocity[env_id])
            self.env_buffers["ground_truth_rays"][env_id].append(ground_truth_rays[env_id])

        self.timestep_count += 1

        # Log progress every 100 timesteps
        if self.timestep_count % 100 == 0:
            total_buffered = sum(len(self.env_buffers["grid_maps"][i]) for i in range(self.num_envs))
            print(
                f"Timestep {self.timestep_count}: {total_buffered} frames buffered, "
                f"{self.frame_idx} frames written, {len(self.episode_boundaries)} episodes completed"
            )

    def end_episodes(self, env_ids: list):
        """
        Mark episodes as complete for specified environment IDs and write them to disk.
        Only processes as many episodes as capacity allows. Exits when full.

        Args:
            env_ids (list): List of environment IDs whose episodes have ended.
        """
        if self.capacity_reached:
            return

        episodes_written = 0
        total_frames_written = 0

        for env_id in env_ids:
            if env_id < 0 or env_id >= self.num_envs:
                raise ValueError(f"Invalid env_id: {env_id}. Must be in range [0, {self.num_envs}).")

            # Get the buffered frames for this episode
            episode_grid_maps = self.env_buffers["grid_maps"][env_id]
            episode_gravity = self.env_buffers["gravity_vectors"][env_id]
            episode_ang_vel = self.env_buffers["angular_velocity"][env_id]
            episode_gt_rays = self.env_buffers["ground_truth_rays"][env_id]

            episode_length = len(episode_grid_maps)

            if episode_length == 0:
                continue  # Skip empty episodes

            # Calculate available space
            available_space = self.num_total_frames - self.frame_idx

            if available_space <= 0:
                # No space left, stop processing more episodes
                self.capacity_reached = True
                break

            # Check if this episode fits
            if episode_length > available_space:
                # Truncate episode to fit available space
                episode_length = available_space
                episode_grid_maps = episode_grid_maps[:episode_length]
                episode_gravity = episode_gravity[:episode_length]
                episode_ang_vel = episode_ang_vel[:episode_length]
                episode_gt_rays = episode_gt_rays[:episode_length]
                print(f"Truncating episode from env {env_id}: {len(episode_grid_maps)} -> {episode_length} frames")

            # Convert lists to numpy arrays
            episode_grid_maps = np.array(episode_grid_maps)
            episode_gravity = np.array(episode_gravity)
            episode_ang_vel = np.array(episode_ang_vel)
            episode_gt_rays = np.array(episode_gt_rays)

            # Record episode boundaries
            start_idx = self.frame_idx
            end_idx = self.frame_idx + episode_length - 1
            self.episode_boundaries.append((start_idx, end_idx))

            # Resize datasets
            new_size = self.frame_idx + episode_length
            self.grid_maps_ds.resize(new_size, axis=0)
            self.gravity_vectors_ds.resize(new_size, axis=0)
            self.angular_velocity_ds.resize(new_size, axis=0)
            self.ground_truth_rays_ds.resize(new_size, axis=0)

            # Write episode as contiguous block
            self.grid_maps_ds[self.frame_idx : new_size] = episode_grid_maps
            self.gravity_vectors_ds[self.frame_idx : new_size] = episode_gravity
            self.angular_velocity_ds[self.frame_idx : new_size] = episode_ang_vel
            self.ground_truth_rays_ds[self.frame_idx : new_size] = episode_gt_rays

            # Update counter
            self.frame_idx = new_size

            # Clear buffer
            self.env_buffers["grid_maps"][env_id] = []
            self.env_buffers["gravity_vectors"][env_id] = []
            self.env_buffers["angular_velocity"][env_id] = []
            self.env_buffers["ground_truth_rays"][env_id] = []

            episodes_written += 1
            total_frames_written += episode_length

            # Check if we've reached capacity
            if self.frame_idx >= self.num_total_frames:
                self.capacity_reached = True
                break

        # Log summary
        if episodes_written > 0:
            print(
                f"Wrote {episodes_written} episode(s) ({total_frames_written} frames) | "
                f"Total: {len(self.episode_boundaries)} episodes, {self.frame_idx} frames "
                f"({100 * self.frame_idx / self.num_total_frames:.1f}%)"
            )

        # If capacity reached, close and exit
        if self.capacity_reached:
            print(f"Dataset capacity reached ({self.num_total_frames} frames).")
            self.close()
            sys.exit(0)

    def close(self):
        """
        Closes the HDF5 file. ONLY writes episode boundaries to disk.
        Does NOT process any buffered episodes.
        """
        if self.file is None:
            return  # Already closed

        # Write episode boundaries to disk
        if len(self.episode_boundaries) > 0:
            self.episode_boundaries_ds.resize(len(self.episode_boundaries), axis=0)
            self.episode_boundaries_ds[:] = np.array(self.episode_boundaries, dtype=np.int64)
            print(f"Wrote {len(self.episode_boundaries)} episode boundaries.")

        # Close file
        self.file.close()
        self.file = None
        print(f"HDF5 file closed. Total: {self.frame_idx} frames, {len(self.episode_boundaries)} episodes")
