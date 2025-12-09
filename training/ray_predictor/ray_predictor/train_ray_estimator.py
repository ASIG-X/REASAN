import argparse
import random
from typing import Dict

import h5py
import numpy as np
import torch
import wandb
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm


# --- U-Net Components ---
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.1),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(0.1),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool1d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = x2.size()[2] - x1.size()[2]
        # This causes the onnx export warnings
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UNet1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


class RayDataset(Dataset):
    """Dataset for loading preprocessed data from H5 files."""

    def __init__(self, h5_file: str, history_frames: int = 10):
        """
        Args:
            h5_file: Path to H5 file containing preprocessed data
            history_frames: Number of consecutive frames to use
        """
        self.h5_file_path = h5_file
        self.history_frames = history_frames

        # Open the H5 file and keep it open for the lifetime of the dataset
        self.h5_file = h5py.File(h5_file, "r")

        # Get data shapes
        self.grid_maps_shape = self.h5_file["grid_maps"].shape  # [N, 10, 30, 180]
        self.gravity_vectors_shape = self.h5_file["gravity_vectors"].shape  # [N, 10, 3]
        self.angular_velocity_shape = self.h5_file["angular_velocity"].shape  # [N, 10, 3]
        self.gt_data_shape = self.h5_file["ground_truth_rays"].shape  # [N, 180]

        # Validate shapes
        assert self.grid_maps_shape[0] == self.gt_data_shape[0], "Grid maps and GT must have same batch size"
        assert self.grid_maps_shape[0] == self.gravity_vectors_shape[0], (
            "Grid maps and gravity must have same batch size"
        )
        assert self.grid_maps_shape[0] == self.angular_velocity_shape[0], (
            "Grid maps and angular velocity must have same batch size"
        )
        assert self.grid_maps_shape[1] >= history_frames, f"Grid maps must have at least {history_frames} frames"
        assert self.gravity_vectors_shape[1] >= history_frames, f"Gravity must have at least {history_frames} frames"
        assert self.angular_velocity_shape[1] >= history_frames, (
            f"Angular velocity must have at least {history_frames} frames"
        )
        assert self.gt_data_shape[1] == 180, "Ground truth must have 180 rays"
        assert self.gravity_vectors_shape[2] == 3, "Gravity data must have 3 components"
        assert self.angular_velocity_shape[2] == 3, "Angular velocity data must have 3 components"
        assert self.grid_maps_shape[2] == 30, "Grid maps must have 30 theta bins (elevation)"
        assert self.grid_maps_shape[3] == 180, "Grid maps must have 180 phi bins (azimuth)"

        self.n_samples = self.grid_maps_shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Load data for this sample directly from the open H5 file
        # Take the last history_frames frames
        grid_maps = self.h5_file["grid_maps"][idx, -self.history_frames :, :, :]  # [history_frames, 30, 180]
        gravity_vectors = self.h5_file["gravity_vectors"][idx, -self.history_frames :, :]  # [history_frames, 3]
        angular_velocity = self.h5_file["angular_velocity"][idx, -self.history_frames :, :]  # [history_frames, 3]
        gt_rays = self.h5_file["ground_truth_rays"][idx]  # [180]

        # Concatenate gravity and angular velocity
        imu_data = np.concatenate([gravity_vectors, angular_velocity], axis=-1)  # [history_frames, 6]

        # Convert to tensors
        grid_maps = torch.from_numpy(grid_maps).float()
        imu_data = torch.from_numpy(imu_data).float()
        gt_rays = torch.from_numpy(gt_rays).float()

        return grid_maps, imu_data, gt_rays

    def __del__(self):
        """Close the H5 file when the dataset is destroyed."""
        if hasattr(self, "h5_file") and self.h5_file is not None:
            self.h5_file.close()


class RayPredictor(nn.Module):
    """Ray predictor using 1D CNN, Transformer, and U-Net."""

    def __init__(
        self,
        history_frames: int = 10,
        grid_width: int = 180,
        grid_height: int = 30,
        cnn_out_channels: int = 16,
        transformer_dim: int = 128,
        transformer_heads: int = 4,
        transformer_layers: int = 3,
        imu_feature_dim: int = 16,
        device: str = "cuda",
    ):
        super().__init__()
        self.history_frames = history_frames
        self.grid_width = grid_width
        self.transformer_dim = transformer_dim

        # 1D CNN to extract spatial features (shared across all frames)
        # Input: (theta_bins=30) -> Output: (cnn_out_channels=16)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=grid_height, out_channels=32, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout1d(0.1),
            nn.Conv1d(in_channels=32, out_channels=cnn_out_channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),
        )

        # IMU encoder - encode 6D IMU to imu_feature_dim per frame
        self.imu_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, imu_feature_dim),
            nn.ReLU(),
        )

        # Temporal encoding - learnable encoding for each time step
        self.temporal_encoding = nn.Parameter(torch.randn(history_frames, cnn_out_channels + imu_feature_dim))

        # Projection layer to transformer dimension
        # Input: (cnn_out_channels + imu_feature_dim) * history_frames
        token_input_dim = (cnn_out_channels + imu_feature_dim) * history_frames
        self.token_projection = nn.Linear(token_input_dim, transformer_dim)

        # Learnable positional encoding for spatial tokens
        self.spatial_pos_encoding = nn.Parameter(torch.randn(grid_width, transformer_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # U-Net for final prediction
        self.unet = UNet1D(in_channels=transformer_dim, out_channels=1)

    def forward(self, grid: torch.Tensor, imu_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model.
        Args:
            grid (torch.Tensor): Shape (B, H, 30, 180) - batch, history, theta, phi
            imu_data (torch.Tensor): Shape (B, H, 6) - batch, history, [gravity(3), angular_velocity(3)]
        Returns:
            torch.Tensor: Shape (B, 180) - predicted ray distances
        """
        batch_size, history, theta_bins, phi_bins = grid.shape

        # Process each frame through the CNN
        # Reshape to (B*H, theta_bins, phi_bins)
        grid_flat = grid.reshape(batch_size * history, theta_bins, phi_bins)

        # Extract spatial features: (B*H, cnn_out, phi_bins)
        spatial_features = self.feature_extractor(grid_flat)

        # Reshape to (B, H, cnn_out, phi_bins)
        spatial_features = spatial_features.reshape(batch_size, history, -1, phi_bins)

        # Encode IMU data for each frame
        # Reshape to (B*H, 6)
        imu_flat = imu_data.reshape(batch_size * history, 6)
        imu_features = self.imu_encoder(imu_flat)  # (B*H, imu_feature_dim)

        # Reshape to (B, H, imu_feature_dim)
        imu_features = imu_features.reshape(batch_size, history, -1)

        # Broadcast IMU features across spatial dimension and concatenate with spatial features
        # imu_features: (B, H, imu_feature_dim) -> (B, H, imu_feature_dim, phi_bins)
        imu_broadcasted = imu_features.unsqueeze(-1).expand(-1, -1, -1, phi_bins)

        # Concatenate: (B, H, cnn_out + imu_feature_dim, phi_bins)
        combined_features = torch.cat([spatial_features, imu_broadcasted], dim=2)

        # Add temporal encoding
        # temporal_encoding: (H, cnn_out + imu_feature_dim)
        # Reshape and broadcast to match combined_features
        temporal_enc = self.temporal_encoding.unsqueeze(0).unsqueeze(-1)  # (1, H, cnn_out + imu_feature_dim, 1)
        combined_features = combined_features + temporal_enc  # Broadcasting

        # Stack features across time to create tokens
        # Permute to (B, phi_bins, H, cnn_out + imu_feature_dim)
        combined_features = combined_features.permute(0, 3, 1, 2)

        # Flatten temporal and feature dimensions: (B, phi_bins, H * (cnn_out + imu_feature_dim))
        tokens = combined_features.reshape(batch_size, phi_bins, -1)

        # Project to transformer dimension: (B, phi_bins, transformer_dim)
        tokens = self.token_projection(tokens)

        # Add spatial positional encoding
        tokens = tokens + self.spatial_pos_encoding.unsqueeze(0)  # Broadcasting

        # Apply transformer: (B, phi_bins, transformer_dim)
        transformer_output = self.transformer(tokens)

        # Permute for U-Net: (B, transformer_dim, phi_bins)
        transformer_output = transformer_output.permute(0, 2, 1)

        # U-Net prediction: (B, 1, phi_bins)
        predicted_rays = self.unet(transformer_output)

        return predicted_rays.squeeze(1)  # (B, phi_bins)


class ConservativeLoss(nn.Module):
    """
    Conservative loss function that penalizes overestimation more than underestimation.
    """

    def __init__(
        self,
        overestimate_penalty=2.0,
        underestimate_penalty=1.0,
        conservative_penalty=0.05,
        conservative_threshold=0.5,
    ):
        super().__init__()
        self.overestimate_penalty = overestimate_penalty
        self.underestimate_penalty = underestimate_penalty
        self.conservative_penalty = conservative_penalty
        self.conservative_threshold = conservative_threshold

    def forward(self, pred, target):
        diff = pred - target
        overestimate_mask = diff > 0
        underestimate_mask = diff <= 0

        huber_loss = F.smooth_l1_loss(pred, target, reduction="none")

        overestimate_loss = huber_loss[overestimate_mask] * self.overestimate_penalty
        underestimate_loss = huber_loss[underestimate_mask] * self.underestimate_penalty

        asymmetric_loss = torch.cat([overestimate_loss, underestimate_loss]).mean()

        short_distance_mask = target < self.conservative_threshold
        conservative_reg = F.relu(pred[short_distance_mask] - target[short_distance_mask] + 0.1).mean()
        conservative_reg *= self.conservative_penalty

        total_loss = asymmetric_loss + conservative_reg
        return total_loss


def create_model(history_frames: int, device: str):
    """
    Factory function to create the model.

    Args:
        history_frames: Number of history frames
        device: Device to place model on

    Returns:
        model: Instantiated model
    """
    return RayPredictor(history_frames=history_frames, device=device).to(device)


def get_loss_function():
    """
    Get the loss function.

    Returns:
        loss_fn: Loss function
        loss_name: Name of the loss function for logging
    """
    return ConservativeLoss(), "ConservativeLoss"


def load_episodes_chunk(h5_file_path: str, episode_indices: list, history_frames: int):
    """
    Load a chunk of complete episodes from sequential H5 file.

    Args:
        h5_file_path: Path to H5 file
        episode_indices: List of episode indices to load
        history_frames: Number of history frames to use

    Returns:
        List of tuples (grid_maps, imu_data, gt_rays, episode_length) for each episode
    """
    episodes_data = []

    with h5py.File(h5_file_path, "r") as f:
        episode_boundaries = f["episode_boundaries"][:]

        for ep_idx in episode_indices:
            start_idx, end_idx = episode_boundaries[ep_idx]
            ep_length = end_idx - start_idx + 1

            # Load all frames for this episode
            grid_maps = f["grid_maps"][start_idx : end_idx + 1]  # [ep_length, 30, 180]
            gravity = f["gravity_vectors"][start_idx : end_idx + 1]  # [ep_length, 3]
            ang_vel = f["angular_velocity"][start_idx : end_idx + 1]  # [ep_length, 3]
            gt_rays = f["ground_truth_rays"][start_idx : end_idx + 1]  # [ep_length, 180]

            # Convert to tensors
            grid_maps = torch.from_numpy(grid_maps).float()
            gravity = torch.from_numpy(gravity).float()
            ang_vel = torch.from_numpy(ang_vel).float()
            gt_rays = torch.from_numpy(gt_rays).float()

            episodes_data.append((grid_maps, gravity, ang_vel, gt_rays, ep_length))

    return episodes_data


def create_training_samples_from_episodes(episodes_data, history_frames):
    """
    Convert episodes to training samples with history.

    Args:
        episodes_data: List of (grid_maps, gravity, ang_vel, gt_rays, ep_length)
        history_frames: Number of history frames

    Returns:
        Tuple of (grid_inputs, imu_inputs, gt_outputs, episode_lengths)
    """
    all_grid_inputs = []
    all_imu_inputs = []
    all_gt_outputs = []
    episode_lengths = []

    for grid_maps, gravity, ang_vel, gt_rays, ep_length in episodes_data:
        # Concatenate IMU data
        imu_data = torch.cat([gravity, ang_vel], dim=-1)  # [ep_length, 6]

        # Create samples for each frame in episode
        ep_grid_samples = []
        ep_imu_samples = []

        for frame_idx in range(ep_length):
            # Determine history window
            hist_start = max(0, frame_idx - history_frames + 1)
            hist_end = frame_idx + 1
            hist_length = hist_end - hist_start

            # Get history data
            hist_grids = grid_maps[hist_start:hist_end]  # [hist_length, 30, 180]
            hist_imu = imu_data[hist_start:hist_end]  # [hist_length, 6]

            # Pad with zeros for grids, repeat first IMU for early frames
            if hist_length < history_frames:
                pad_length = history_frames - hist_length

                # Zero padding for grids
                zero_grids = torch.zeros(pad_length, 30, 180)
                hist_grids = torch.cat([zero_grids, hist_grids], dim=0)

                # Repeat first IMU
                first_imu = hist_imu[0:1].expand(pad_length, -1)
                hist_imu = torch.cat([first_imu, hist_imu], dim=0)

            ep_grid_samples.append(hist_grids)
            ep_imu_samples.append(hist_imu)

        # Stack samples for this episode
        ep_grid_samples = torch.stack(ep_grid_samples, dim=0)  # [ep_length, history_frames, 30, 180]
        ep_imu_samples = torch.stack(ep_imu_samples, dim=0)  # [ep_length, history_frames, 6]

        all_grid_inputs.append(ep_grid_samples)
        all_imu_inputs.append(ep_imu_samples)
        all_gt_outputs.append(gt_rays)
        episode_lengths.append(ep_length)

    # Concatenate all episodes
    grid_inputs = torch.cat(all_grid_inputs, dim=0)  # [total_frames, history_frames, 30, 180]
    imu_inputs = torch.cat(all_imu_inputs, dim=0)  # [total_frames, history_frames, 6]
    gt_outputs = torch.cat(all_gt_outputs, dim=0)  # [total_frames, 180]

    return grid_inputs, imu_inputs, gt_outputs, episode_lengths


def train_model(
    h5_file: str,
    val_h5_file: str = None,
    history_frames: int = None,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = "cuda",
):
    """
    Train the ray predictor model using chunk-based loading from H5 files.
    """
    # Get dataset info and auto-detect history_frames if not provided
    with h5py.File(h5_file, "r") as f:
        episode_boundaries = f["episode_boundaries"][:]
        total_episodes = episode_boundaries.shape[0]
        total_frames = f["grid_maps"].shape[0]
        print(f"Detected sequential format: {total_episodes} episodes, {total_frames} frames")

    # Use provided history_frames or auto-detect from data
    if history_frames is None:
        raise ValueError("history_frames must be specified for sequential data format")
    print(f"Using provided history frames: {history_frames}")

    print(f"Using batch size: {batch_size}, num_epochs: {num_epochs}, learning_rate: {learning_rate}")

    # Initialize wandb
    wandb.login()
    wandb.init(
        project="ray_predictor",
        config={
            "history_frames": history_frames,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
        },
    )

    # Validation dataset setup
    val_total_episodes = 0
    if val_h5_file:
        with h5py.File(val_h5_file, "r") as f:
            val_total_episodes = f["episode_boundaries"].shape[0]

    # Create model
    model = create_model(history_frames, device)
    print("Created RayPredictor model")

    # Log model architecture
    wandb.watch(model, log_freq=100)

    # Loss function and optimizer
    criterion, loss_name = get_loss_function()
    print(f"Using loss function: {loss_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Initialize training state
    start_epoch = 0
    best_val_loss = float("inf")
    global_step = 0

    # Training loop
    # Create progress bar for epochs starting from start_epoch
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        # Create shuffled chunk order for this epoch
        episode_indices = list(range(total_episodes))
        random.shuffle(episode_indices)

        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        batch_episodes = []
        batch_frame_count = 0

        train_pbar = tqdm(episode_indices, desc=f"Epoch {epoch + 1}/{num_epochs} Training", leave=False)

        # Process chunks in shuffled order
        for ep_idx in train_pbar:
            with h5py.File(h5_file, "r") as f:
                start_idx, end_idx = f["episode_boundaries"][ep_idx]
                ep_length = end_idx - start_idx + 1

            batch_episodes.append(ep_idx)
            batch_frame_count += ep_length

            if batch_frame_count >= batch_size:
                ep_data = load_episodes_chunk(h5_file, batch_episodes, history_frames)
                grid_inputs, imu_inputs, gt_outputs, _ = create_training_samples_from_episodes(ep_data, history_frames)

                grid_inputs = grid_inputs.to(device)
                imu_inputs = imu_inputs.to(device)
                gt_outputs = gt_outputs.to(device)

                # Normalize input data by dividing by 3 (max range)
                grid_inputs = grid_inputs / 3.0

                # Forward pass
                optimizer.zero_grad()
                predicted_rays = model(grid_inputs, imu_inputs)

                # Compute loss
                loss = criterion(predicted_rays, gt_outputs)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_batches += 1
                global_step += 1

                # Update progress bar with current loss
                train_pbar.set_postfix(loss=f"{loss.item():.6f}")

                # Log per-batch training loss to wandb
                wandb.log(
                    {
                        "train_loss_batch": loss.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "global_step": global_step,
                        "batch_frames": batch_frame_count,
                    },
                    step=global_step,
                )

                batch_episodes = []
                batch_frame_count = 0

                del grid_inputs, imu_inputs, gt_outputs, predicted_rays, ep_data
                torch.cuda.empty_cache()

        # Process remaining episodes if any
        if len(batch_episodes) > 0:
            ep_data = load_episodes_chunk(h5_file, batch_episodes, history_frames)
            grid_inputs, imu_inputs, gt_outputs, episode_lengths = create_training_samples_from_episodes(
                ep_data, history_frames
            )

            grid_inputs = grid_inputs.to(device)
            imu_inputs = imu_inputs.to(device)
            gt_outputs = gt_outputs.to(device)
            grid_inputs = grid_inputs / 3.0

            optimizer.zero_grad()
            predictions = model(grid_inputs, imu_inputs)
            loss = criterion(predictions, gt_outputs)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            global_step += 1

            del grid_inputs, imu_inputs, gt_outputs, predictions, ep_data
            torch.cuda.empty_cache()

        train_pbar.close()
        avg_train_loss = train_loss / train_batches

        # Validation phase (per epoch)
        val_loss = 0.0
        if val_h5_file:
            model.eval()
            val_batches = 0

            with torch.no_grad():
                val_episode_indices = list(range(val_total_episodes))
                batch_episodes = []
                batch_frame_count = 0

                val_pbar = tqdm(val_episode_indices, desc=f"Epoch {epoch + 1}/{num_epochs} Validation", leave=False)

                for ep_idx in val_pbar:
                    with h5py.File(val_h5_file, "r") as f:
                        start_idx, end_idx = f["episode_boundaries"][ep_idx]
                        ep_length = end_idx - start_idx + 1

                    batch_episodes.append(ep_idx)
                    batch_frame_count += ep_length

                    if batch_frame_count >= batch_size:
                        ep_data = load_episodes_chunk(val_h5_file, batch_episodes, history_frames)
                        grid_inputs, imu_inputs, gt_outputs, _ = create_training_samples_from_episodes(
                            ep_data, history_frames
                        )

                        grid_inputs = grid_inputs.to(device)
                        imu_inputs = imu_inputs.to(device)
                        gt_outputs = gt_outputs.to(device)

                        grid_inputs = grid_inputs / 3.0

                        predicted_rays = model(grid_inputs, imu_inputs)

                        loss = criterion(predicted_rays, gt_outputs)
                        val_loss += loss.item()
                        val_batches += 1

                        val_pbar.set_postfix(loss=f"{loss.item():.6f}")

                        batch_episodes = []
                        batch_frame_count = 0

                        del grid_inputs, imu_inputs, gt_outputs, predicted_rays, ep_data
                        torch.cuda.empty_cache()

                if len(batch_episodes) > 0:
                    ep_data = load_episodes_chunk(val_h5_file, batch_episodes, history_frames)
                    grid_inputs, imu_inputs, gt_outputs, _ = create_training_samples_from_episodes(
                        ep_data, history_frames
                    )

                    grid_inputs = grid_inputs.to(device)
                    imu_inputs = imu_inputs.to(device)
                    gt_outputs = gt_outputs.to(device)

                    grid_inputs = grid_inputs / 3.0

                    predicted_rays = model(grid_inputs, imu_inputs)

                    loss = criterion(predicted_rays, gt_outputs)
                    val_loss += loss.item()
                    val_batches += 1

                    del grid_inputs, imu_inputs, gt_outputs, predicted_rays, ep_data
                    torch.cuda.empty_cache()

                val_pbar.close()
                avg_val_loss = val_loss / val_batches

                # Track best validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    # Save best model
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "train_loss": avg_train_loss,
                            "val_loss": avg_val_loss,
                            "best_val_loss": best_val_loss,
                        },
                        "ray_predictor_best.pth",
                    )

        # Update learning rate scheduler
        scheduler.step()

        # Log epoch-level metrics
        log_dict = {
            "epoch": epoch,
            "train_loss_epoch": avg_train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        if val_h5_file:
            log_dict["val_loss_epoch"] = avg_val_loss
            log_dict["best_val_loss"] = best_val_loss

        wandb.log(log_dict, step=global_step)

        # Update epoch progress bar with summary
        postfix_dict = {"train_loss": f"{avg_train_loss:.6f}"}
        if val_h5_file:
            postfix_dict["val_loss"] = f"{avg_val_loss:.6f}"
            postfix_dict["best_val"] = f"{best_val_loss:.6f}"
        epoch_pbar.set_postfix(postfix_dict)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
            }
            if val_h5_file:
                checkpoint_dict["val_loss"] = avg_val_loss

            torch.save(checkpoint_dict, f"ray_predictor_checkpoint_epoch_{epoch + 1}.pth")

    # Close progress bars
    epoch_pbar.close()

    # Save final model
    final_dict = {
        "epoch": num_epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
    }
    if val_h5_file:
        final_dict["val_loss"] = avg_val_loss

    torch.save(final_dict, "ray_predictor_final.pth")

    print("Training completed!")
    print(f"Final training loss: {avg_train_loss:.6f}")
    if val_h5_file:
        print(f"Best validation loss: {best_val_loss:.6f}")

    # Finish wandb run
    wandb.finish()


def predict_model(
    h5_file: str,
    checkpoint_path: str,
    output_file: str,
    history_frames: int = None,
    batch_size: int = 512,
    device: str = "cuda",
):
    """
    Generate predictions using a trained model with chunk-based loading from H5 files.
    """
    # Get dataset info and auto-detect history_frames if not provided
    with h5py.File(h5_file, "r") as f:
        episode_boundaries = f["episode_boundaries"][:]
        total_episodes = episode_boundaries.shape[0]
        total_frames = f["grid_maps"].shape[0]

    # Use provided history_frames or auto-detect from data
    if history_frames is None:
        raise ValueError("history_frames must be specified for sequential data format")
    print(f"Using provided history frames: {history_frames}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = create_model(history_frames, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Generate predictions
    all_predictions = []
    all_losses = []
    criterion, loss_name = get_loss_function()
    print(f"Using loss function: {loss_name}")

    with torch.no_grad():
        episode_indices = list(range(total_episodes))
        batch_episodes = []
        batch_frame_count = 0

        pred_pbar = tqdm(episode_indices, desc="Generating Predictions")

        for ep_idx in pred_pbar:
            with h5py.File(h5_file, "r") as f:
                start_idx, end_idx = f["episode_boundaries"][ep_idx]
                ep_length = end_idx - start_idx + 1

            batch_episodes.append(ep_idx)
            batch_frame_count += ep_length

            if batch_frame_count >= batch_size:
                ep_data = load_episodes_chunk(h5_file, batch_episodes, history_frames)
                grid_inputs, imu_inputs, gt_outputs, _ = create_training_samples_from_episodes(ep_data, history_frames)

                grid_inputs = grid_inputs.to(device)
                imu_inputs = imu_inputs.to(device)
                gt_outputs = gt_outputs.to(device)

                grid_inputs = grid_inputs / 3.0

                predictions = model(grid_inputs, imu_inputs)
                loss = criterion(predictions, gt_outputs)

                all_losses.append(loss.item())
                all_predictions.append(predictions.cpu().numpy())

                pred_pbar.set_postfix(loss=f"{loss.item():.6f}")

                batch_episodes = []
                batch_frame_count = 0

                del grid_inputs, imu_inputs, gt_outputs, predictions, ep_data
                torch.cuda.empty_cache()

        if len(batch_episodes) > 0:
            ep_data = load_episodes_chunk(h5_file, batch_episodes, history_frames)
            grid_inputs, imu_inputs, gt_outputs, _ = create_training_samples_from_episodes(ep_data, history_frames)

            grid_inputs = grid_inputs.to(device)
            imu_inputs = imu_inputs.to(device)
            gt_outputs = gt_outputs.to(device)

            grid_inputs = grid_inputs / 3.0

            predictions = model(grid_inputs, imu_inputs)
            loss = criterion(predictions, gt_outputs)

            all_losses.append(loss.item())
            all_predictions.append(predictions.cpu().numpy())

            del grid_inputs, imu_inputs, gt_outputs, predictions, ep_data
            torch.cuda.empty_cache()

        pred_pbar.close()

        predictions = np.concatenate(all_predictions, axis=0) if all_predictions else np.array([])

        # Save predictions to H5 file
        with h5py.File(output_file, "w") as f:
            f.create_dataset("pred_rays", data=predictions)

        print(f"Saved predictions to {output_file}")
        print(f"Prediction shape: {predictions.shape}")
        print(f"H5 file contains dataset 'pred_rays' with shape {predictions.shape}")

        # Print loss statistics
        if all_losses:
            losses_np = np.array(all_losses)
            print("\n--- Evaluation Loss Statistics (per batch) ---")
            print(f"  Average Loss: {losses_np.mean():.6f}")
            print(f"  Min Loss:     {losses_np.min():.6f}")
            print(f"  Max Loss:     {losses_np.max():.6f}")
            print("-------------------------------------------------")


def export_model(
    checkpoint_path: str,
    output_path: str,
    export_format: str,
    history_frames: int = 10,
    grid_width: int = 180,
    grid_height: int = 30,
    device: str = "cuda",
):
    """
    Export a trained model to TorchScript or ONNX format.
    """
    print(f"Exporting model to {export_format.upper()} format...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = create_model(history_frames, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    example_grid = torch.randn(1, history_frames, grid_height, grid_width, device=device)
    example_imu = torch.randn(1, history_frames, 6, device=device)

    if export_format.lower() == "torchscript":
        try:
            print("Attempting to export using torch.jit.script...")
            scripted_model = torch.jit.script(model)
            scripted_model.save(output_path)
            print(f"Successfully exported TorchScript model to: {output_path}")

            print("Verifying exported model...")
            loaded_model = torch.jit.load(output_path, map_location=device)
            with torch.no_grad():
                output = loaded_model(example_grid, example_imu)
                print(f"Verification successful! Output shape: {output.shape}")

        except Exception as e:
            print(f"torch.jit.script failed: {e}")
            print("Attempting to export using torch.jit.trace...")
            try:
                with torch.no_grad():
                    traced_model = torch.jit.trace(model, (example_grid, example_imu))
                    traced_model.save(output_path)
                    print(f"Successfully exported traced TorchScript model to: {output_path}")

                    print("Verifying exported model...")
                    loaded_model = torch.jit.load(output_path, map_location=device)
                    output = loaded_model(example_grid, example_imu)
                    print(f"Verification successful! Output shape: {output.shape}")

            except Exception as e2:
                print(f"torch.jit.trace also failed: {e2}")
                raise RuntimeError("Both torch.jit.script and torch.jit.trace failed")

    elif export_format.lower() == "onnx":
        try:
            import onnx

            print("Exporting to ONNX format...")

            model.eval()

            torch.onnx.export(
                model,
                (example_grid, example_imu),
                output_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["grid", "imu_data"],
                output_names=["predicted_rays"],
                dynamic_axes={},
            )
            print(f"Successfully exported ONNX model to: {output_path}")

            print("Verifying exported ONNX model...")
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model verification successful!")

            try:
                import onnxruntime as ort

                print("Testing with ONNX Runtime...")

                ort_session = ort.InferenceSession(output_path)

                ort_inputs = {
                    "grid": example_grid.cpu().numpy(),
                    "imu_data": example_imu.cpu().numpy(),
                }

                ort_outputs = ort_session.run(None, ort_inputs)
                print(f"ONNX Runtime test successful! Output shape: {ort_outputs[0].shape}")

            except ImportError:
                print("ONNX Runtime not available, skipping runtime verification")
            except Exception as e:
                print(f"ONNX Runtime test failed: {e}")

        except ImportError:
            raise ImportError("ONNX package not found. Please install with: pip install onnx")
        except Exception as e:
            raise RuntimeError(f"ONNX export failed: {e}")

    else:
        raise ValueError(f"Unsupported export format: {export_format}. Use 'torchscript' or 'onnx'")

    print("Model export completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or predict with Ray Predictor Baselines")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict", "export"],
        default="train",
        help="Mode: train, predict, or export",
    )
    parser.add_argument("--h5_file", type=str, help="Path to training H5 file")
    parser.add_argument("--val_h5_file", type=str, help="Path to validation H5 file")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint file to load (required for prediction and export)")
    parser.add_argument("--output_file", type=str, help="Output file for predictions or exported model")
    parser.add_argument(
        "--export_format", type=str, choices=["torchscript", "onnx"], help="Export format (required for export mode)"
    )
    parser.add_argument(
        "--history_frames", type=int, default=None, help="Number of history frames (auto-detected if not provided)"
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    if args.mode == "train":
        if not args.h5_file:
            parser.error("--h5_file is required for training mode")

        train_model(
            h5_file=args.h5_file,
            val_h5_file=args.val_h5_file,
            history_frames=args.history_frames,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            device=args.device,
        )
    elif args.mode == "predict":
        if not args.checkpoint:
            parser.error("--checkpoint is required for prediction mode")
        if not args.output_file:
            parser.error("--output_file is required for prediction mode")
        if not args.h5_file:
            parser.error("--h5_file is required for prediction mode")

        predict_model(
            h5_file=args.h5_file,
            checkpoint_path=args.checkpoint,
            output_file=args.output_file,
            history_frames=args.history_frames,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.mode == "export":
        if not args.checkpoint:
            parser.error("--checkpoint is required for export mode")
        if not args.output_file:
            parser.error("--output_file is required for export mode")
        if not args.export_format:
            parser.error("--export_format is required for export mode")

        export_model(
            checkpoint_path=args.checkpoint,
            output_path=args.output_file,
            export_format=args.export_format,
            history_frames=args.history_frames,
            device=args.device,
        )
