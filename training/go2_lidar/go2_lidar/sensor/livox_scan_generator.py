import os

import numpy as np
import torch


class LivoxScanGenerator:
    """
    parallel livox lidar scan pattern generator.
    code adapted from https://github.com/aCodeDog/OmniPerception.
    """

    livox_lidar_params = {
        "avia": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 70.4,
            "vertical_fov": 77.2,
            "samples": 24000,
        },
        "HAP": {"laser_min_range": 0.1, "laser_max_range": 200.0, "samples": 45300, "downsample": 1},
        "horizon": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 81.7,
            "vertical_fov": 25.1,
            "samples": 24000,
        },
        "mid40": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 81.7,
            "vertical_fov": 25.1,
            "samples": 24000,
        },
        "mid70": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 70.4,
            "vertical_fov": 70.4,
            "samples": 10000,
        },
        "mid360": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            # assuming 20hz lidar and 200000 points/sec
            # check here: https://terra-1-g.djicdn.com/851d20f7b9f64838a34cd02351370894/Livox/Livox_Mid-360_User_Manual_EN.pdf#page=22.06
            "samples": 6000,
        },
        "tele": {
            "laser_min_range": 0.1,
            "laser_max_range": 200.0,
            "horizontal_fov": 14.5,
            "vertical_fov": 16.1,
            "samples": 24000,
        },
    }

    def __init__(self, name, num_envs, device):
        if name in self.livox_lidar_params:
            self.device = device
            self.num_envs = num_envs
            self.laser_min_range = self.livox_lidar_params[name]["laser_min_range"]
            self.laser_max_range = self.livox_lidar_params[name]["laser_max_range"]
            self.n_samples = self.livox_lidar_params[name]["samples"]
            self.mode = 0
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                scan_mode_file = os.path.join(script_dir, "scan_mode", f"{name}.npy")
                self.ray_angles = np.load(scan_mode_file)
                self.ray_angles = torch.from_numpy(self.ray_angles).to(device)
            except FileNotFoundError:
                raise FileNotFoundError("Scan mode file not found!")
            self.n_rays = self.ray_angles.shape[0]
        else:
            raise ValueError(f"Invalid LiDAR name: {name}")

        # index per env
        self.curr_start_index = torch.zeros(self.num_envs, dtype=torch.long, device=device) + 6000

    @property
    def num_rays(self):
        return self.n_samples

    def reset_index(self, env_ids=None):
        if env_ids is None:
            self.curr_start_index[:] = 0
        else:
            self.curr_start_index[env_ids] = 0

    def sample_ray_angles(self, env_ids=None, downsample=1):
        """
        Sample theta and phi angles for rays per env.

        Return value: torch.Tensor of shape (self.num_envs, self.n_samples, 2).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        sample_offsets = torch.arange(self.n_samples, device=self.device)
        indices = self.curr_start_index[env_ids].unsqueeze(1) + sample_offsets.unsqueeze(0)
        wrapped_indices = indices % self.n_rays
        ray_out = self.ray_angles[wrapped_indices]
        self.curr_start_index[env_ids] = (self.curr_start_index[env_ids] + self.n_samples) % self.n_rays
        if downsample > 1:
            if self.n_samples % downsample != 0:
                raise ValueError("Downsampling factor must divide the number of samples evenly.")
            ray_out = ray_out[:, ::downsample, :]
        return ray_out

    def sample_rays(self, env_ids=None):
        """
        Sample ray_starts and ray_directions.

        Return values:
        - ray_starts: (self.num_envs, self.n_samples, 3);
        - ray_dirs: (self.num_envs, self.n_samples, 3).
        """
        downsample = 1

        original_n_samples = self.n_samples
        rand = np.random.random()
        if rand < 0.1:
            if self.mode == 1 or self.mode == 2:
                self.mode = 0
            elif rand < 0.08:
                self.mode = 1
            else:
                self.mode = 2

        self.mode = 0

        if self.mode == 0:
            ray_out = self.sample_ray_angles(env_ids=env_ids, downsample=downsample)
        elif self.mode == 1:
            self.n_samples = 3000
            ray_out = self.sample_ray_angles(env_ids=env_ids, downsample=downsample)
            ray_out = torch.cat([ray_out, ray_out], dim=1)
            self.n_samples = original_n_samples
        else:
            self.n_samples = 2000
            ray_out = self.sample_ray_angles(env_ids=env_ids, downsample=downsample)
            ray_out = torch.cat([ray_out, ray_out, ray_out], dim=1)
            self.n_samples = original_n_samples

        cos_phi = torch.cos(ray_out[:, :, 1])
        sin_phi = torch.sin(ray_out[:, :, 1])
        cos_theta = torch.cos(ray_out[:, :, 0])
        sin_theta = torch.sin(ray_out[:, :, 0])

        ray_dirs = torch.zeros(ray_out.shape[0], self.n_samples // downsample, 3, device=self.device)
        ray_dirs[:, :, 0] = cos_phi * cos_theta
        ray_dirs[:, :, 1] = cos_phi * sin_theta
        ray_dirs[:, :, 2] = sin_phi

        ray_starts = torch.zeros_like(ray_dirs, device=self.device)

        return ray_starts, ray_dirs
