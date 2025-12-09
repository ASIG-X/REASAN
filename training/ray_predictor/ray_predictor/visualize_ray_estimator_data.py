import argparse
import os
from typing import Optional

import h5py
import numpy as np
import plotly.graph_objects as go
import torch
from nicegui import ui
from torch import nn


class LidarDataVisualizer:
    def __init__(self, h5_file: str, pred_h5_file: Optional[str] = None):
        self.h5_file = h5_file
        self.pred_h5_file = pred_h5_file

        # Get data shapes and validate without loading all data
        print("Reading H5 file metadata...")
        with h5py.File(h5_file, "r") as f:
            # Check if this is sequential format or history-based format
            if "episode_boundaries" in f:
                # Sequential format
                self.is_sequential = True
                self.episode_boundaries = f["episode_boundaries"][:]  # Load all boundaries (small dataset)
                print(f"Detected sequential format with {len(self.episode_boundaries)} episodes")

                # Get shapes for sequential format
                grid_maps_shape = f["grid_maps"].shape  # [total_frames, 30, 180]
                gravity_vectors_shape = f["gravity_vectors"].shape  # [total_frames, 3]
                angular_velocity_shape = f["angular_velocity"].shape  # [total_frames, 3]
                gt_data_shape = f["ground_truth_rays"].shape  # [total_frames, 180]

                self.total_frames = grid_maps_shape[0]
                self.history_frames = 1  # We'll handle history differently for sequential data

                # Get metadata
                if "num_envs" in f.attrs:
                    self.num_envs = f.attrs["num_envs"]
                else:
                    self.num_envs = 1  # Default if not specified

            else:
                # History-based format (original)
                self.is_sequential = False
                self.episode_boundaries = None

                # Get shapes without loading data
                grid_maps_shape = f["grid_maps"].shape  # [N, history_frames, 30, 180]
                gravity_vectors_shape = f["gravity_vectors"].shape  # [N, history_frames, 3]
                angular_velocity_shape = f["angular_velocity"].shape  # [N, history_frames, 3]
                gt_data_shape = f["ground_truth_rays"].shape  # [N, 180]

                # Extract history frames from the data shape
                self.history_frames = grid_maps_shape[1]
                self.total_frames = grid_maps_shape[0]

            # Check if predictions are available in the same H5 file
            self.has_pred_rays = "pred_rays" in f
            if self.has_pred_rays:
                pred_rays_shape = f["pred_rays"].shape
                print(f"Found prediction rays in main H5 file with shape: {pred_rays_shape}")
                if self.is_sequential:
                    assert pred_rays_shape[0] == self.total_frames, "Prediction rays must have same number of frames"
                else:
                    assert pred_rays_shape == gt_data_shape, "Prediction rays shape must match GT rays shape"

        # Check for external prediction H5 file
        self.has_external_pred = False
        if pred_h5_file:
            print(f"Reading external prediction H5 file: {pred_h5_file}")
            with h5py.File(pred_h5_file, "r") as f:
                if "pred_rays" in f:
                    external_pred_shape = f["pred_rays"].shape
                    print(f"Found prediction rays in external H5 file with shape: {external_pred_shape}")
                    if self.is_sequential:
                        assert external_pred_shape[0] == self.total_frames, (
                            "External prediction rays must have same number of frames"
                        )
                    else:
                        assert external_pred_shape == gt_data_shape, (
                            "External prediction rays shape must match GT rays shape"
                        )
                    self.has_external_pred = True
                else:
                    print("Warning: External H5 file does not contain 'pred_rays' dataset")

        print(f"Grid maps shape: {grid_maps_shape}")
        print(f"Gravity vectors shape: {gravity_vectors_shape}")
        print(f"Angular velocity shape: {angular_velocity_shape}")
        print(f"Ground truth rays shape: {gt_data_shape}")

        if not self.is_sequential:
            # Validate data shapes for history-based format
            assert len(grid_maps_shape) == 4, "Grid maps should be 4D"
            assert grid_maps_shape[1] > 0, "Second dimension should have at least 1 history frame"
            print(f"Detected history frames: {self.history_frames}")

        # Common validation
        assert grid_maps_shape[-2] == 30, "Second-to-last dimension should be 30 (phi bins, vertical)"
        assert grid_maps_shape[-1] == 180, "Last dimension should be 180 (theta bins, horizontal)"

        if self.is_sequential:
            self.n_samples = self.total_frames
            # For sequential data, we'll navigate by frame, not sample
            self.current_episode = 0
            self.current_frame_in_episode = 0
        else:
            self.n_samples = gt_data_shape[0]
            # No sorting - use original order
            self.sorted_indices = np.arange(self.n_samples)
            self.losses = None

        # Generate ray directions starting from -X and going counter-clockwise
        # Start at 180 degrees (-X direction) and go 360 degrees without overlap
        angles = np.linspace(-np.pi, np.pi - np.pi / 180.0, 180, endpoint=False)
        self.ray_directions = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(180)])

        # UI elements
        self.plot = None
        self.sample_slider = None
        self.frame_slider = None
        self.show_rays_toggle = None
        self.sample_info_label = None

        self.current_sample = 0
        self.current_frame = self.history_frames - 1 if not self.is_sequential else 0
        self.show_rays = False
        self.show_pred_rays = False  # New toggle for prediction rays
        self.show_grid = True  # New toggle for spherical grid (default: show)
        self.show_gravity = True  # New toggle for gravity vector (default: show)
        self.show_angular_velocity = True  # New toggle for angular velocity (default: show)

        # Add camera state tracking with default camera position
        # Store camera state for perspective mode
        self.perspective_camera_state = dict(
            eye=dict(x=1.25, y=1.25, z=1.25), center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)
        )

        # Add plot type tracking (True = 2D top-down, False = 3D perspective)
        self.is_2d_view = False

        # Add UI revision tracking
        self.uirevision = "constant"

    def load_grid_map(self, sample_idx: int, frame_idx: int = None) -> np.ndarray:
        """Load a specific grid map on-the-fly"""
        with h5py.File(self.h5_file, "r") as f:
            if self.is_sequential:
                # For sequential format, sample_idx is the absolute frame index
                return f["grid_maps"][sample_idx]
            else:
                # For history-based format
                return f["grid_maps"][sample_idx, frame_idx]

    def load_gt_rays(self, sample_idx: int) -> np.ndarray:
        """Load ground truth rays for a specific sample on-the-fly"""
        with h5py.File(self.h5_file, "r") as f:
            return f["ground_truth_rays"][sample_idx]

    def load_pred_rays(self, sample_idx: int) -> Optional[np.ndarray]:
        """Load prediction rays for a specific sample on-the-fly"""
        if self.has_external_pred:
            with h5py.File(self.pred_h5_file, "r") as f:
                return f["pred_rays"][sample_idx]
        elif self.has_pred_rays:
            with h5py.File(self.h5_file, "r") as f:
                return f["pred_rays"][sample_idx]
        else:
            return None

    def load_gravity_vector(self, sample_idx: int, frame_idx: int = None) -> np.ndarray:
        """Load gravity vector for a specific sample and frame on-the-fly"""
        with h5py.File(self.h5_file, "r") as f:
            if self.is_sequential:
                return f["gravity_vectors"][sample_idx]
            else:
                return f["gravity_vectors"][sample_idx, frame_idx]

    def load_angular_velocity(self, sample_idx: int, frame_idx: int = None) -> np.ndarray:
        """Load angular velocity for a specific sample and frame on-the-fly"""
        with h5py.File(self.h5_file, "r") as f:
            if self.is_sequential:
                return f["angular_velocity"][sample_idx]
            else:
                return f["angular_velocity"][sample_idx, frame_idx]

    def create_grid_visualization(self, grid: np.ndarray) -> go.Scatter3d:
        """
        Create a 3D scatter plot of the spherical grid.
        Each grid cell is represented as a point with color indicating distance.
        Grid shape: [30, 180] = [phi_bins, theta_bins]
        """
        phi_bins, theta_bins = grid.shape  # [30, 180]

        # Create coordinate grids for visualization
        # Theta: -180° to 180° (horizontal, azimuth)
        theta_angles = np.linspace(-np.pi, np.pi, theta_bins, endpoint=False)  # 180 points
        # Phi: -5° to 55° (vertical, elevation)
        phi_angles = np.linspace(np.deg2rad(-5), np.deg2rad(55), phi_bins)  # 30 points

        # Create meshgrid and flatten for scatter plot
        # Use indexing='ij' to match the grid array layout
        phi_mesh, theta_mesh = np.meshgrid(phi_angles, theta_angles, indexing="ij")
        theta_flat = theta_mesh.flatten()
        phi_flat = phi_mesh.flatten()
        distance_flat = grid.flatten()

        # Convert to Cartesian coordinates
        x_points = distance_flat * np.cos(phi_flat) * np.cos(theta_flat)
        y_points = distance_flat * np.cos(phi_flat) * np.sin(theta_flat)
        z_points = distance_flat * np.sin(phi_flat)

        # Filter out points at max_range (empty cells)
        valid_mask = distance_flat < 3.01  # Just below max_range of 3.0

        scatter = go.Scatter3d(
            x=x_points[valid_mask],
            y=y_points[valid_mask],
            z=z_points[valid_mask],
            mode="markers",
            marker=dict(
                size=2,
                color=distance_flat[valid_mask],
                colorscale="Plasma",
                opacity=0.6,
            ),
            name="Spherical Grid",
            hovertemplate="Distance: %{marker.color:.2f}m<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>",
        )

        return scatter

    def create_gravity_visualization(self, gravity_vector: np.ndarray) -> go.Scatter3d:
        """
        Create a 3D arrow visualization for the gravity vector.
        Gravity vector shape: [3] = [x, y, z]
        """
        # Scale the gravity vector for better visualization (gravity magnitude is ~9.81)
        # Normalize and scale to a reasonable length for visualization
        gravity_norm = np.linalg.norm(gravity_vector)
        if gravity_norm > 0:
            # Scale to length of 1.0 for visualization
            scaled_gravity = gravity_vector / gravity_norm
        else:
            scaled_gravity = np.array([0, 0, -1])  # Default downward if zero

        # Create arrow from origin to gravity direction
        arrow_trace = go.Scatter3d(
            x=[0, scaled_gravity[0]],
            y=[0, scaled_gravity[1]],
            z=[0, scaled_gravity[2]],
            mode="lines+markers",
            line=dict(color="blue", width=8),
            marker=dict(
                size=[4, 12],  # Smaller at origin, larger at tip
                color="blue",
                symbol=["circle", "diamond"],
            ),
            name="Gravity Direction",
            hovertemplate=f"Gravity Vector<br>X: {gravity_vector[0]:.2f}<br>Y: {gravity_vector[1]:.2f}<br>Z: {gravity_vector[2]:.2f}<br>Magnitude: {gravity_norm:.2f}<extra></extra>",
        )

        return arrow_trace

    def create_plot(self) -> go.Figure:
        """Create the 3D plotly figure"""
        if self.is_2d_view:
            return self.create_2d_plot()
        else:
            return self.create_3d_plot()

    def create_3d_plot(self) -> go.Figure:
        """Create the 3D perspective plotly figure"""
        fig = go.Figure()

        if self.is_sequential:
            frame_index = self.current_sample
        else:
            original_index = self.get_original_index()
            frame_index = original_index

        # Add spherical grid if enabled
        if self.show_grid:
            # Load the grid for current sample and frame on-the-fly
            if self.is_sequential:
                grid = self.load_grid_map(frame_index)
            else:
                grid = self.load_grid_map(self.get_original_index(), self.current_frame)
            grid_surface = self.create_grid_visualization(grid)
            fig.add_trace(grid_surface)

        # Add gravity vector if enabled
        if self.show_gravity:
            # Load gravity vector for current sample and frame on-the-fly
            if self.is_sequential:
                gravity_vector = self.load_gravity_vector(frame_index)
            else:
                gravity_vector = self.load_gravity_vector(self.get_original_index(), self.current_frame)
            gravity_trace = self.create_gravity_visualization(gravity_vector)
            fig.add_trace(gravity_trace)

        # Add ground truth rays if enabled (as connected line + dots)
        if self.show_rays:
            # Load GT rays for current sample on-the-fly
            gt_distances = self.load_gt_rays(frame_index)
            gt_distances *= 3.0  # Scale to match grid range

            # Calculate ray endpoints
            ray_ends = self.ray_directions * gt_distances[:, np.newaxis]

            # Add line connecting all ray endpoints (creates a closed loop)
            fig.add_trace(
                go.Scatter3d(
                    x=np.append(ray_ends[:, 0], ray_ends[0, 0]),  # Close the loop
                    y=np.append(ray_ends[:, 1], ray_ends[0, 1]),
                    z=np.append(ray_ends[:, 2], ray_ends[0, 2]),
                    mode="lines",
                    line=dict(color="#E74C3C", width=4),
                    opacity=0.6,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Add all 180 rays as scatter points (on top of lines)
            fig.add_trace(
                go.Scatter3d(
                    x=ray_ends[:, 0],
                    y=ray_ends[:, 1],
                    z=ray_ends[:, 2],
                    mode="markers",
                    marker=dict(
                        size=2,
                        color="#E74C3C",  # Soft red
                        symbol="circle",
                        opacity=0.8,
                    ),
                    name="GT Rays",
                    hovertemplate="GT Ray %{pointNumber}<br>Angle: %{customdata:.0f}°<br>Distance: %{text:.2f}m<extra></extra>",
                    customdata=np.arange(180) * 2,  # Angles in degrees
                    text=gt_distances,  # Distances for hover
                )
            )

        # Add prediction rays if enabled and data is available (as connected line + dots)
        if self.show_pred_rays and (self.has_external_pred or self.has_pred_rays):
            pred_distances = self.load_pred_rays(frame_index)
            pred_distances *= 3.0  # Scale to match grid range

            # Calculate ray endpoints
            pred_ray_ends = self.ray_directions * pred_distances[:, np.newaxis]

            # Add line connecting all ray endpoints (creates a closed loop)
            fig.add_trace(
                go.Scatter3d(
                    x=np.append(pred_ray_ends[:, 0], pred_ray_ends[0, 0]),  # Close the loop
                    y=np.append(pred_ray_ends[:, 1], pred_ray_ends[0, 1]),
                    z=np.append(pred_ray_ends[:, 2], pred_ray_ends[0, 2]),
                    mode="lines",
                    line=dict(color="#2ECC71", width=4),
                    opacity=0.6,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Add all 180 prediction rays as scatter points (on top of lines)
            fig.add_trace(
                go.Scatter3d(
                    x=pred_ray_ends[:, 0],
                    y=pred_ray_ends[:, 1],
                    z=pred_ray_ends[:, 2],
                    mode="markers",
                    marker=dict(
                        size=2,
                        color="#2ECC71",  # Soft green
                        symbol="circle",
                        opacity=0.8,
                    ),
                    name="Pred Rays",
                    hovertemplate="Pred Ray %{pointNumber}<br>Angle: %{customdata:.0f}°<br>Distance: %{text:.2f}m<extra></extra>",
                    customdata=np.arange(180) * 2,  # Angles in degrees
                    text=pred_distances,  # Distances for hover
                )
            )

        # Use fixed axis ranges based on the 6-meter clipping in data collection
        axis_range = 4.0
        axis_limits = [-axis_range, axis_range]

        # Update layout with fixed axis ranges centered on origin
        if self.is_sequential:
            title_text = f"Frame: {self.current_sample}/{self.n_samples - 1}"
        else:
            title_text = f"Sample: {self.current_sample}, Frame: {self.current_frame}"
            if self.losses is not None:
                title_text += f", Loss: {self.losses[self.current_sample]:.4f}"

        layout_kwargs = dict(
            title=title_text,
            scene=dict(
                xaxis=dict(title="X", range=axis_limits),
                yaxis=dict(title="Y", range=axis_limits),
                zaxis=dict(title="Z", range=axis_limits),
                aspectmode="cube",
                aspectratio=dict(x=1, y=1, z=1),
                camera=dict(**self.perspective_camera_state, projection=dict(type="perspective")),
            ),
            width=1200,
            height=800,
            uirevision=self.uirevision,  # Preserve UI state including camera position
        )

        fig.update_layout(**layout_kwargs)

        return fig

    def create_2d_plot(self) -> go.Figure:
        """Create a 2D top-down plotly figure (X-Y plane view)"""
        fig = go.Figure()

        if self.is_sequential:
            frame_index = self.current_sample
        else:
            original_index = self.get_original_index()
            frame_index = original_index

        # Add spherical grid if enabled (project to X-Y plane)
        if self.show_grid:
            if self.is_sequential:
                grid = self.load_grid_map(frame_index)
            else:
                grid = self.load_grid_map(original_index, self.current_frame)

            phi_bins, theta_bins = grid.shape
            theta_angles = np.linspace(-np.pi, np.pi, theta_bins, endpoint=False)
            phi_angles = np.linspace(np.deg2rad(-5), np.deg2rad(55), phi_bins)

            phi_mesh, theta_mesh = np.meshgrid(phi_angles, theta_angles, indexing="ij")
            theta_flat = theta_mesh.flatten()
            phi_flat = phi_mesh.flatten()
            distance_flat = grid.flatten()

            # Convert to Cartesian coordinates (X-Y projection)
            x_points = distance_flat * np.cos(phi_flat) * np.cos(theta_flat)
            y_points = distance_flat * np.cos(phi_flat) * np.sin(theta_flat)

            # Filter out points at max_range
            valid_mask = distance_flat < 3.01

            fig.add_trace(
                go.Scatter(
                    x=x_points[valid_mask],
                    y=y_points[valid_mask],
                    mode="markers",
                    marker=dict(
                        size=3,
                        color=distance_flat[valid_mask],
                        colorscale="Plasma",
                        opacity=0.6,
                    ),
                    name="Spherical Grid",
                    hovertemplate="Distance: %{marker.color:.2f}m<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
                )
            )

        # Add gravity vector projection if enabled
        if self.show_gravity:
            if self.is_sequential:
                gravity_vector = self.load_gravity_vector(frame_index)
            else:
                gravity_vector = self.load_gravity_vector(original_index, self.current_frame)

            gravity_norm = np.linalg.norm(gravity_vector)
            if gravity_norm > 0:
                scaled_gravity = gravity_vector / gravity_norm
            else:
                scaled_gravity = np.array([0, 0, -1])

            # Project to X-Y plane
            fig.add_trace(
                go.Scatter(
                    x=[0, scaled_gravity[0]],
                    y=[0, scaled_gravity[1]],
                    mode="lines+markers",
                    line=dict(color="blue", width=4),
                    marker=dict(size=[8, 16], color="blue"),
                    name="Gravity (X-Y)",
                    hovertemplate=f"Gravity X-Y Projection<extra></extra>",
                )
            )

        # Add angular velocity visualization if enabled (only in 2D view)
        if self.show_angular_velocity:
            if self.is_sequential:
                angular_velocity = self.load_angular_velocity(frame_index)
            else:
                angular_velocity = self.load_angular_velocity(original_index, self.current_frame)
            angular_velocity_traces = self.create_angular_velocity_visualization_2d(angular_velocity)
            for trace in angular_velocity_traces:
                fig.add_trace(trace)

        # Add ground truth rays if enabled
        if self.show_rays:
            gt_distances = self.load_gt_rays(frame_index)
            gt_distances *= 3.0  # Scale to match grid range
            ray_ends = self.ray_directions * gt_distances[:, np.newaxis]

            # Add closed loop line
            fig.add_trace(
                go.Scatter(
                    x=np.append(ray_ends[:, 0], ray_ends[0, 0]),
                    y=np.append(ray_ends[:, 1], ray_ends[0, 1]),
                    mode="lines",
                    line=dict(color="#E74C3C", width=3),
                    opacity=0.6,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Add scatter points
            fig.add_trace(
                go.Scatter(
                    x=ray_ends[:, 0],
                    y=ray_ends[:, 1],
                    mode="markers",
                    marker=dict(size=4, color="#E74C3C", opacity=0.8),
                    name="GT Rays",
                    hovertemplate="GT Ray %{pointNumber}<br>Angle: %{customdata:.0f}°<br>Distance: %{text:.2f}m<extra></extra>",
                    customdata=np.arange(180) * 2,
                    text=gt_distances,
                )
            )

        # Add prediction rays if enabled
        if self.show_pred_rays and (self.has_external_pred or self.has_pred_rays):
            pred_distances = self.load_pred_rays(frame_index)
            pred_distances *= 3.0  # Scale to match grid range
            pred_ray_ends = self.ray_directions * pred_distances[:, np.newaxis]

            # Add closed loop line
            fig.add_trace(
                go.Scatter(
                    x=np.append(pred_ray_ends[:, 0], pred_ray_ends[0, 0]),
                    y=np.append(pred_ray_ends[:, 1], pred_ray_ends[0, 1]),
                    mode="lines",
                    line=dict(color="#2ECC71", width=3),
                    opacity=0.6,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Add scatter points
            fig.add_trace(
                go.Scatter(
                    x=pred_ray_ends[:, 0],
                    y=pred_ray_ends[:, 1],
                    mode="markers",
                    marker=dict(size=4, color="#2ECC71", opacity=0.8),
                    name="Pred Rays",
                    hovertemplate="Pred Ray %{pointNumber}<br>Angle: %{customdata:.0f}°<br>Distance: %{text:.2f}m<extra></extra>",
                    customdata=np.arange(180) * 2,
                    text=pred_distances,
                )
            )

        # Use fixed axis ranges
        axis_range = 4.0
        axis_limits = [-axis_range, axis_range]

        # Update layout for 2D plot
        if self.is_sequential:
            title_text = f"2D Top-Down View | Frame: {self.current_sample}/{self.n_samples - 1}"
        else:
            title_text = f"2D Top-Down View | Sample: {self.current_sample}, Frame: {self.current_frame}"
            if self.losses is not None:
                title_text += f", Loss: {self.losses[self.current_sample]:.4f}"

        fig.update_layout(
            title=title_text,
            xaxis=dict(title="X", range=axis_limits, scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Y", range=axis_limits),
            width=1200,
            height=800,
            uirevision=self.uirevision,
        )

        return fig

    def create_angular_velocity_visualization_2d(self, angular_velocity: np.ndarray) -> list:
        """
        Create a 2D arc visualization for angular velocity in the X-Y plane.
        Angular velocity shape: [3] = [x, y, z] (we'll focus on z-component for 2D rotation)
        Returns a list of traces for the arc and arrow.
        """
        # Get the z-component of angular velocity (rotation around Z-axis)
        omega_z = angular_velocity[2]

        # If angular velocity is too small, don't visualize
        if abs(omega_z) < 0.01:
            return []

        # Create an arc to show rotation direction and magnitude
        # Arc radius and angle based on angular velocity magnitude
        arc_radius = 0.8  # Fixed radius for visualization
        arc_angle = np.clip(abs(omega_z) * 30, 5, 90)  # Scale to degrees, clamp between 5 and 90

        # Determine rotation direction (counter-clockwise for positive, clockwise for negative)
        if omega_z > 0:
            # Counter-clockwise rotation
            start_angle = 0
            end_angle = np.deg2rad(arc_angle)
        else:
            # Clockwise rotation
            start_angle = 0
            end_angle = -np.deg2rad(arc_angle)

        # Generate arc points
        num_arc_points = 30
        arc_angles = np.linspace(start_angle, end_angle, num_arc_points)
        arc_x = arc_radius * np.cos(arc_angles)
        arc_y = arc_radius * np.sin(arc_angles)

        traces = []

        # Add arc line
        traces.append(
            go.Scatter(
                x=arc_x,
                y=arc_y,
                mode="lines",
                line=dict(color="orange", width=3),
                name="Angular Velocity",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Add arrow at the end of the arc using a proper triangular arrow
        arrow_tip_x = arc_x[-1]
        arrow_tip_y = arc_y[-1]

        # Calculate tangent direction at the end of the arc
        tangent_angle = end_angle + (np.pi / 2 if omega_z > 0 else -np.pi / 2)

        # Create arrow head as a small triangle pointing in the tangent direction
        arrow_size = 0.08
        # Arrow head vertices relative to the tip
        arrow_vertices_x = [
            0,  # tip
            -arrow_size * np.cos(tangent_angle) + arrow_size * 0.5 * np.cos(tangent_angle + np.pi / 2),  # left
            -arrow_size * np.cos(tangent_angle) + arrow_size * 0.5 * np.cos(tangent_angle - np.pi / 2),  # right
            0,  # back to tip to close triangle
        ]
        arrow_vertices_y = [
            0,  # tip
            -arrow_size * np.sin(tangent_angle) + arrow_size * 0.5 * np.sin(tangent_angle + np.pi / 2),  # left
            -arrow_size * np.sin(tangent_angle) + arrow_size * 0.5 * np.sin(tangent_angle - np.pi / 2),  # right
            0,  # back to tip to close triangle
        ]

        # Translate arrow to the actual position
        arrow_x = [arrow_tip_x + dx for dx in arrow_vertices_x]
        arrow_y = [arrow_tip_y + dy for dy in arrow_vertices_y]

        # Add arrow head as a filled shape
        traces.append(
            go.Scatter(
                x=arrow_x,
                y=arrow_y,
                mode="lines",
                line=dict(color="orange", width=2),
                fill="toself",
                fillcolor="orange",
                showlegend=False,
                hovertemplate=(
                    f"Angular Velocity (Z)<br>"
                    f"ω_x: {angular_velocity[0]:.2f} rad/s<br>"
                    f"ω_y: {angular_velocity[1]:.2f} rad/s<br>"
                    f"ω_z: {angular_velocity[2]:.2f} rad/s<extra></extra>"
                ),
            )
        )

        # Add a label showing the magnitude
        traces.append(
            go.Scatter(
                x=[arc_radius * 0.6 * np.cos(end_angle / 2)],
                y=[arc_radius * 0.6 * np.sin(end_angle / 2)],
                mode="text",
                text=[f"ω_z: {omega_z:.2f}"],
                textposition="middle center",
                textfont=dict(size=10, color="orange"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        return traces

    def get_original_index(self) -> int:
        """Get the original data index from the current sorted sample index."""
        if self.is_sequential:
            return self.get_current_frame_index()
        return self.sorted_indices[self.current_sample] if hasattr(self, "sorted_indices") else self.current_sample

    def get_current_frame_index(self) -> int:
        """Get the absolute frame index in the sequential dataset"""
        if not self.is_sequential:
            return self.get_original_index()

        # For sequential data, just return the current sample index
        # With the new contiguous storage, frames are stored sequentially within episodes
        return self.current_sample

    def update_plot(self):
        """Update the plot with current settings"""
        if self.plot:
            fig = self.create_plot()
            self.plot.update_figure(fig)

    def on_sample_change(self, value: int):
        """Handle sample slider change"""
        self.current_sample = value
        self.update_sample_info_label()
        self.update_plot()

    def on_frame_change(self, value: int):
        """Handle frame slider change"""
        self.current_frame = value
        self.update_plot()

    def on_rays_toggle(self, value: bool):
        """Handle rays visibility toggle"""
        self.show_rays = value
        self.update_plot()

    def on_pred_rays_toggle(self, value: bool):
        """Handle prediction rays visibility toggle"""
        self.show_pred_rays = value
        self.update_plot()

    def on_grid_toggle(self, value: bool):
        """Handle spherical grid visibility toggle"""
        self.show_grid = value
        self.update_plot()

    def on_gravity_toggle(self, value: bool):
        """Handle gravity vector visibility toggle"""
        self.show_gravity = value
        self.update_plot()

    def on_angular_velocity_toggle(self, value: bool):
        """Handle angular velocity visibility toggle"""
        self.show_angular_velocity = value
        self.update_plot()

    def on_episode_change(self, value: int):
        """Handle episode change for sequential format"""
        self.current_episode = value
        # Reset frame in episode to 0 when changing episodes
        self.current_frame_in_episode = 0

        # Update frame slider max value for new episode
        if hasattr(self, "frame_in_episode_slider"):
            if self.current_episode < len(self.episode_boundaries):
                start_idx, end_idx = self.episode_boundaries[self.current_episode]
                max_frames = (end_idx - start_idx) // self.num_envs
                self.frame_in_episode_slider._props["max"] = max_frames
                self.frame_in_episode_slider.value = 0

        self.update_episode_info_label()
        self.update_plot()

    def on_frame_in_episode_change(self, value: int):
        """Handle frame change within episode for sequential format"""
        self.current_frame_in_episode = value
        self.update_plot()

    def on_plot_type_toggle(self, value: bool):
        """Handle plot type toggle (3D perspective / 2D top-down)"""
        # Store current camera state if in 3D mode
        if not self.is_2d_view and self.plot and hasattr(self.plot, "figure") and "scene" in self.plot.figure.layout:
            current_camera = self.plot.figure.layout.scene.camera
            self.perspective_camera_state = dict(
                eye=dict(x=current_camera.eye.x, y=current_camera.eye.y, z=current_camera.eye.z),
                center=dict(x=current_camera.center.x, y=current_camera.center.y, z=current_camera.center.z),
                up=dict(x=current_camera.up.x, y=current_camera.up.y, z=current_camera.up.z),
            )

        # Switch plot type
        self.is_2d_view = value
        self.update_plot()

    @property
    def camera_state(self):
        """Get current camera state based on projection mode"""
        return self.perspective_camera_state

    def create_ui(self):
        """Create the NiceGUI interface"""
        ui.label("Lidar Training Data Visualizer").classes("text-2xl font-bold mb-4 text-center w-full")

        with ui.row().classes("w-full no-wrap items-start"):
            # --- Controls Column (Sidebar) ---
            with ui.column().classes("w-1/4 gap-4 p-4 border rounded"):
                ui.label("Controls").classes("text-lg font-semibold")

                ui.separator()

                # Simple frame/sample navigation for both formats
                if self.is_sequential:
                    ui.label("Frame")
                else:
                    ui.label("Sample")

                with ui.row().classes("w-full items-center no-wrap"):
                    ui.button(icon="fast_rewind", on_click=lambda: self.navigate_sample(-100)).props("flat round dense")
                    ui.button(icon="arrow_left", on_click=lambda: self.navigate_sample(-1)).props("flat round dense")
                    self.sample_slider = ui.slider(
                        min=0,
                        max=self.n_samples - 1,
                        value=self.current_sample,
                        on_change=lambda e: self.on_sample_change(int(e.value)),
                    ).classes("flex-grow")
                    ui.button(icon="arrow_right", on_click=lambda: self.navigate_sample(1)).props("flat round dense")
                    ui.button(icon="fast_forward", on_click=lambda: self.navigate_sample(100)).props("flat round dense")

                self.sample_info_label = ui.label()
                self.update_sample_info_label()

                if not self.is_sequential:
                    ui.separator()

                    ui.label("Frame")
                    with ui.row().classes("w-full items-center no-wrap"):
                        ui.button(icon="fast_rewind", on_click=lambda: self.navigate_frame(-5)).props(
                            "flat round dense"
                        )
                        ui.button(icon="arrow_left", on_click=lambda: self.navigate_frame(-1)).props("flat round dense")
                        self.frame_slider = ui.slider(
                            min=0,
                            max=self.history_frames - 1,
                            value=self.current_frame,
                            on_change=lambda e: self.on_frame_change(int(e.value)),
                        ).classes("flex-grow")
                        ui.button(icon="arrow_right", on_click=lambda: self.navigate_frame(1)).props("flat round dense")
                        ui.button(icon="fast_forward", on_click=lambda: self.navigate_frame(5)).props(
                            "flat round dense"
                        )

                    ui.label().bind_text_from(
                        self.frame_slider,
                        "value",
                        lambda v: f"Frame: {v}/{self.history_frames - 1} {'(Latest)' if v == self.history_frames - 1 else ''}",
                    )

                ui.separator()

                self.show_rays_toggle = ui.checkbox(
                    "Show Ground Truth Rays", value=self.show_rays, on_change=lambda e: self.on_rays_toggle(e.value)
                )

                if self.has_external_pred or self.has_pred_rays:
                    self.show_pred_rays_toggle = ui.checkbox(
                        "Show Prediction Rays",
                        value=self.show_pred_rays,
                        on_change=lambda e: self.on_pred_rays_toggle(e.value),
                    )

                self.show_grid_toggle = ui.checkbox(
                    "Show Spherical Grid", value=self.show_grid, on_change=lambda e: self.on_grid_toggle(e.value)
                )

                self.show_gravity_toggle = ui.checkbox(
                    "Show Gravity Vector", value=self.show_gravity, on_change=lambda e: self.on_gravity_toggle(e.value)
                )

                self.show_angular_velocity_toggle = ui.checkbox(
                    "Show Angular Velocity (2D only)",
                    value=self.show_angular_velocity,
                    on_change=lambda e: self.on_angular_velocity_toggle(e.value),
                )

                ui.separator()

                # Plot type toggle
                self.plot_type_toggle = ui.checkbox(
                    "2D Top-Down View",
                    value=self.is_2d_view,
                    on_change=lambda e: self.on_plot_type_toggle(e.value),
                )

                ui.separator()

                # Info
                if self.is_sequential:
                    info_text = f"Sequential format: {self.total_frames} total frames, {len(self.episode_boundaries)} total episodes"
                    if self.num_envs > 1:
                        info_text += f", {self.num_envs} parallel envs"
                else:
                    info_text = f"History format: {self.n_samples} samples, {self.history_frames} frames, 180 GT rays"

                if self.has_external_pred:
                    info_text += ", predictions loaded (external H5 file)"
                elif self.has_pred_rays:
                    info_text += ", predictions loaded (main H5 file)"
                info_text += ", IMU data (gravity + angular velocity)"
                ui.label(info_text).classes("text-xs text-gray-500")

            # --- Plot Column ---
            with ui.column().classes("w-3/4"):
                self.plot = ui.plotly(self.create_plot()).classes("w-full h-full")

    def navigate_sample(self, offset: int):
        """Navigate to the previous/next sample in the sorted list."""
        new_sample_idx = self.current_sample + offset
        if 0 <= new_sample_idx < self.n_samples:
            # Setting the slider's value will automatically trigger its on_change handler
            if self.sample_slider:
                self.sample_slider.value = new_sample_idx

    def navigate_frame(self, offset: int):
        """Navigate to the previous/next frame."""
        new_frame_idx = self.current_frame + offset
        if 0 <= new_frame_idx < self.history_frames:
            if self.frame_slider:
                self.frame_slider.value = new_frame_idx

    def navigate_episode(self, offset: int):
        """Navigate to the previous/next episode in sequential format."""
        new_episode_idx = self.current_episode + offset
        if 0 <= new_episode_idx < len(self.episode_boundaries):
            if hasattr(self, "episode_slider"):
                self.episode_slider.value = new_episode_idx

    def navigate_frame_in_episode(self, offset: int):
        """Navigate to the previous/next frame within the current episode."""
        if self.current_episode >= len(self.episode_boundaries):
            return

        start_idx, end_idx = self.episode_boundaries[self.current_episode]
        max_frames = (end_idx - start_idx) // self.num_envs + 1

        new_frame_idx = self.current_frame_in_episode + offset
        if 0 <= new_frame_idx < max_frames:
            if hasattr(self, "frame_in_episode_slider"):
                self.frame_in_episode_slider.value = new_frame_idx

    def update_sample_info_label(self):
        """Update the UI label with detailed sample information."""
        if hasattr(self, "sample_info_label") and self.sample_info_label:
            if self.is_sequential:
                # Find which episode the current frame belongs to
                current_episode = -1
                for i, (start, end) in enumerate(self.episode_boundaries):
                    if start <= self.current_sample <= end:
                        current_episode = i
                        frame_in_episode = self.current_sample - start
                        episode_length = end - start + 1
                        break

                if current_episode >= 0:
                    text = f"Frame: {self.current_sample}/{self.n_samples - 1} | Episode: {current_episode}/{len(self.episode_boundaries) - 1} | Frame in Ep: {frame_in_episode}/{episode_length - 1}"
                else:
                    text = f"Frame: {self.current_sample}/{self.n_samples - 1}"
            else:
                text = f"Sample: {self.current_sample}/{self.n_samples - 1}"
                if self.losses is not None:
                    text += f" | Loss: {self.losses[self.current_sample]:.4f}"
            self.sample_info_label.set_text(text)

    def update_episode_info_label(self):
        """Update the UI label with episode information for sequential format."""
        if self.is_sequential and hasattr(self, "episode_info_label") and self.episode_info_label:
            if self.current_episode < len(self.episode_boundaries):
                start_idx, end_idx = self.episode_boundaries[self.current_episode]
                episode_length = (end_idx - start_idx) // self.num_envs + 1
                text = f"Episode {self.current_episode}/{len(self.episode_boundaries) - 1} | Length: {episode_length} frames"
            else:
                text = f"Episode {self.current_episode} (invalid)"
            self.episode_info_label.set_text(text)


parser = argparse.ArgumentParser(description="Visualize lidar training data")
parser.add_argument("--input", "-i", required=True, help="Path to input H5 data file (.h5)")
parser.add_argument("--pred", "-pr", help="Path to prediction H5 file (.h5) containing 'pred_rays' dataset")
parser.add_argument("--port", "-p", type=int, default=8080, help="Port for web interface")
parser.add_argument("--host", default="localhost", help="Host for web interface")

args = parser.parse_args()

# Validate files exist
if not os.path.exists(args.input):
    raise FileNotFoundError(f"Input file not found: {args.input}")
if args.pred and not os.path.exists(args.pred):
    raise FileNotFoundError(f"Prediction file not found: {args.pred}")

# Create visualizer
visualizer = LidarDataVisualizer(args.input, args.pred)

# Create UI
visualizer.create_ui()

# Run the app
ui.run(host=args.host, port=args.port, title="Lidar Data Visualizer")
