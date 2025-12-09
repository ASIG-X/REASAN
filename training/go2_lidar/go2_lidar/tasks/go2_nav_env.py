# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import asyncio
import copy
import math
import os

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import (
    Articulation,
    RigidObject,
    RigidObjectCfg,
)
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
    RED_ARROW_X_MARKER_CFG,
)
from isaaclab.sensors import (
    ContactSensor,
)
from isaaclab.terrains import TerrainImporter

from go2_lidar.sensor.raycaster_dynamic import RayCasterDynamic
from go2_lidar.tasks.go2_nav_env_cfg import Go2NavEnvCfg
from go2_lidar.utils.hdf5_data_collector import HDF5DatasetWriter_General, HDF5DatasetWriter_Ray


class Go2NavEnv(DirectRLEnv):
    cfg: Go2NavEnvCfg

    def __init__(self, cfg: Go2NavEnvCfg, render_mode: str | None = None, **kwargs):
        print("Initializing training environment...")
        super().__init__(cfg, render_mode, **kwargs)

        if self.cfg.use_dynamic_obstacle:
            print("Using dynamic obstacles in the environment.")
        else:
            print("Not using dynamic obstacles in the environment.")
        self._wait_for_key()

        print(f"number of dynamic obstacles: {self._num_obstacles}")
        self._wait_for_key()

        loco_policy_path = "./logs/rsl_rl/go2_lidar/loco_1/exported/policy.pt"
        print(f"Loading locomotion policy from: {loco_policy_path}")
        self._loco_policy = torch.jit.load(loco_policy_path)
        self._loco_policy.to(self.device).eval()
        hidden_state_shape = self._loco_policy.hidden_state.shape
        cell_state_shape = self._loco_policy.cell_state.shape
        self._loco_policy.hidden_state = torch.zeros(
            hidden_state_shape[0],
            self.num_envs,
            hidden_state_shape[2],
            device=self.device,
        )
        self._loco_policy.cell_state = torch.zeros(
            cell_state_shape[0],
            self.num_envs,
            cell_state_shape[2],
            device=self.device,
        )
        for param in self._loco_policy.parameters():
            param.requires_grad = False
        self._wait_for_key()

        filter_policy_path = "./logs/rsl_rl/go2_lidar/filter_1/exported/policy.pt"
        print(f"Loading filter policy from: {filter_policy_path}")
        self._filter_policy = torch.jit.load(filter_policy_path)
        self._filter_policy.to(self.device).eval()
        hidden_state_shape = self._filter_policy.h_state.shape
        cell_state_shape = self._filter_policy.c_state.shape
        self._filter_policy.h_state = torch.zeros(
            hidden_state_shape[0],
            self.num_envs,
            hidden_state_shape[2],
            device=self.device,
        )
        self._filter_policy.c_state = torch.zeros(
            cell_state_shape[0],
            self.num_envs,
            cell_state_shape[2],
            device=self.device,
        )
        for param in self._filter_policy.parameters():
            param.requires_grad = False
        self._wait_for_key()

        self._ray_predictor = None
        ray_predictor_path = "./go2_lidar/ray_predictor/ray_predictor.pt"
        print(f"Loading ray estimator model from: {ray_predictor_path}")
        try:
            self._ray_predictor = torch.jit.load(ray_predictor_path)
            self._ray_predictor.to(self.device).eval()
            print("Ray estimator model loaded:")
            print(self._ray_predictor)
        except Exception:
            print("Failed to load ray estimator model.")
        self._wait_for_key()

        print(f"num ray centers: {self.cfg.num_ray_centers}")
        self._wait_for_key()

        self._randomize_mass()

        self._reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._timeout_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._init_physx_material_buffer()
        self._reset_physx_materials(torch.ones(self.num_envs, device="cpu", dtype=torch.bool))

        for _, v in self._robot.actuators.items():
            v.stiffness[:] = (torch.rand_like(v.stiffness) * 0.2 + 0.9) * 35.0
            v.damping[:] = (torch.rand_like(v.damping) * 0.2 + 0.9) * 0.5

        self._first_reset = True

        self._current_grid = None

        self._last_dist_to_goal = torch.zeros(self.num_envs, device=self.device)

        self._high_actions = torch.zeros(self.num_envs, self.cfg.num_high_actions, device=self.device)
        self._prev_high_actions = [self._high_actions.clone()] * 5
        self._num_high_actions = self._high_actions.shape[1]

        self._filter_actions = torch.zeros(self.num_envs, 3, device=self.device)

        self._loco_actions = torch.zeros(self.num_envs, self.cfg.num_loco_actions, device=self.device)
        self._prev_loco_actions = [self._loco_actions.clone()] * 5
        self._num_loco_actions = self._loco_actions.shape[1]

        self._obs_buf = torch.zeros(self.num_envs, self.cfg.observation_space, device=self.device)
        self._obs_actor_rays = torch.zeros(self.num_envs, 180, device=self.device)

        self._init_data_extra_buf()

        self._base_id_cs, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids_cs, _ = self._contact_sensor.find_bodies(".*foot")
        self._undesired_contact_body_ids_cs, _ = self._contact_sensor.find_bodies(
            [
                ".*thigh",
                ".*calf",
                "base",
                ".*hip",
                "Head.*",
            ]
        )
        self._feet_ids_bd, _ = self._robot.find_bodies(".*foot")
        self._hip_ids_jt, _ = self._robot.find_joints(".*hip.*")

        self._episode_sums = {}
        self._step_counter = 0
        self._iter_counter = 0

        self._track_env_id = 0
        self._track_env_curriculum = self.num_terrain_rows - 1
        self._track_camera_height = 5.0
        self._reset_tracked_env = False

        self._cmd_goal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._cmd_goal_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._cmd_goal_dir_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._cmd_goal_heading_b = torch.zeros(self.num_envs, 1, device=self.device)

        self._goal_dist_range = torch.zeros(self.num_envs, 2, device=self.device)
        self._goal_dist_range[:, 0] = 1.0
        self._goal_dist_range[:, 1] = 5.0

        self._obst_dist_range = torch.zeros(self.num_envs, 2, device=self.device)
        self._obst_dist_range[:, 0] = 2.0
        self._obst_dist_range[:, 1] = 5.0

        self._vel_cmd_limits = torch.tensor([[2.5, 1.5, 3.0]], device=self.device)

        self._no_obstacle_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._obst_pos_xy_a = torch.zeros(self.num_envs, self._num_obstacles, 2, device=self.device)
        self._obst_pos_xy_b = torch.zeros(self.num_envs, self._num_obstacles, 2, device=self.device)
        self._obst_speed = torch.zeros(self.num_envs, self._num_obstacles, 1, device=self.device)
        self._obst_speed_range = self.cfg.obst_speed_range
        print(f"obstacle speed range: {self._obst_speed_range}")
        self._wait_for_key()

        self._episode_length_sec = 15.0
        self._episode_length_sec_per_env = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_length_sec_per_env[:] = self._episode_length_sec

        self._ray_directions_b = torch.zeros(self.num_envs, 180, 3, device=self.device)
        self._ttc_full = torch.ones(self.num_envs, 180, device=self.device) * 3.0

        self._guiding_vectors = torch.zeros(self.num_envs, 2, device=self.device)
        self._goal_idx_general = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._goal_idx_dead_end = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._goal_idx_detour = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._proximal_ray_dist = 3.0
        self._num_prev_data = 15
        self._prev_grid_buf = torch.ones(self.num_envs, self._num_prev_data, 30, 180, device=self.device) * 3.0
        self._prev_proj_gravity_buf = torch.zeros(self.num_envs, self._num_prev_data, 3, device=self.device)
        self._prev_ang_vel_buf = torch.zeros(self.num_envs, self._num_prev_data, 3, device=self.device)

        self._init_data_collection()
        self._wait_for_key()

        self.border_x = self.cfg.terrain.terrain_generator.num_rows * self.cfg.terrain.terrain_generator.size[0] * 0.5
        self.border_y = self.cfg.terrain.terrain_generator.num_cols * self.cfg.terrain.terrain_generator.size[1] * 0.5

        self._env_type = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._env_type[: self.num_envs // 2] = 1
        self._env_type[self.num_envs // 2 :] = 3

        self._flat_patches_1 = self._terrain.flat_patches["general_start"].clone()
        self._flat_patches_1_goals = self._terrain.flat_patches["general_goal"].clone()
        self._flat_patches_2 = self._terrain.flat_patches["dead_end_start"].clone()
        self._flat_patches_2_goals = self._terrain.flat_patches["dead_end_goal"].clone()
        self._flat_patches_3 = self._terrain.flat_patches["detour_start"].clone()
        self._flat_patches_3_goals = self._terrain.flat_patches["detour_goal"].clone()
        print("Flat patches for curriculum prepared.")
        print(f"patches 1 shape: {self._flat_patches_1.shape}")
        print(f"patches 1 goals shape: {self._flat_patches_1_goals.shape}")
        print(f"patches 2 shape: {self._flat_patches_2.shape}")
        print(f"patches 2 goals shape: {self._flat_patches_2_goals.shape}")
        print(f"patches 3 shape: {self._flat_patches_3.shape}")
        print(f"patches 3 goals shape: {self._flat_patches_3_goals.shape}")
        self._wait_for_key()

        self._general_env_guide = self._terrain.flat_patches["general_guide_generator"]
        print(f"general guide class: {self._general_env_guide}")
        self._wait_for_key()

        self._dead_end_env_guide = self._terrain.flat_patches["dead_end_guide_generator"]
        print(f"dead end guide class: {self._dead_end_env_guide}")
        self._wait_for_key()

        self._detour_env_guide = self._terrain.flat_patches["detour_guide_generator"]
        print(f"detour guide class: {self._detour_env_guide}")
        self._wait_for_key()

        self._robot_start_pos = torch.zeros(self.num_envs, 3, device=self.device)

        if self.cfg.use_predicted_rays:
            print("=" * 50)
            print("Using predicted rays for policy input.")
            print("=" * 50)
            self._wait_for_key()

        self._init_debug_viz()
        self._update_debug_draw()

        asyncio.ensure_future(self.setup_ui())

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.num_terrain_rows = self.cfg.terrain.terrain_generator.num_rows
        self.num_terrain_cols = self.cfg.terrain.terrain_generator.num_cols

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain: TerrainImporter = self.cfg.terrain.class_type(self.cfg.terrain)

        rand_cols = torch.randint(0, self.num_terrain_cols, size=(self.num_envs,), device=self.device)
        if self.cfg.is_play_env:
            rand_cols = (torch.arange(self.num_envs, device=self.device)) % self.num_terrain_cols
        self._env_terrain_cols = rand_cols
        self._terrain.env_origins[:] = self._terrain.terrain_origins[0, rand_cols]
        self._terrain.terrain_levels[:] = 0

        if self.cfg.use_dynamic_obstacle:
            self._num_obstacles = 9
        else:
            self._num_obstacles = 0
        self._create_obstacles()

        self._raycaster = RayCasterDynamic(self.cfg.raycaster, sim_mid360=True)
        self._raycaster_measure = RayCasterDynamic(self.cfg.raycaster_measure, sim_mid360=False)
        self.scene.sensors["raycaster"] = self._raycaster
        self.scene.sensors["raycaster_measure"] = self._raycaster_measure

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        sky_light_cfg = sim_utils.DomeLightCfg(intensity=5000.0)
        sky_light_cfg.func("/World/skyLight", sky_light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        if actions.shape[1] != self.cfg.num_high_actions:
            raise ValueError(f"Expected number of actions {self.cfg.num_high_actions}, but got {actions.shape}")
        self._step_counter += 1
        self._iter_counter = self._step_counter // 24

        self._update_debug_draw()

        if self._step_counter < 1e10 and self.cfg.is_play_env and self.viewport_camera_controller is not None:
            lookat_pos = self._robot.data.root_pos_w[self._track_env_id].cpu()
            eye_pos = lookat_pos.clone()
            eye_pos[0] += 1.0
            eye_pos[1] += 1.0
            eye_pos[2] += self._track_camera_height
            self.viewport_camera_controller.update_view_location(eye=eye_pos, lookat=lookat_pos)

        self._prev_high_actions.append(self._high_actions.clone())
        self._prev_high_actions.pop(0)
        self._high_actions = actions.clone()

        self._cmd_goal_b = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self._robot.data.root_quat_w), self._cmd_goal_w - self._robot.data.root_pos_w
        )
        cmd_goal_dir_norm = torch.norm(self._cmd_goal_b, dim=-1)
        self._cmd_goal_dir_b = self._cmd_goal_b / (cmd_goal_dir_norm[:, None] + 1e-6)
        self._cmd_goal_dir_b[cmd_goal_dir_norm < 0.1] = 0.0
        self._cmd_goal_heading_b = torch.atan2(self._cmd_goal_dir_b[:, 1], self._cmd_goal_dir_b[:, 0]).unsqueeze(-1)

        filter_proprio_obs = torch.cat(
            [
                self._robot.data.root_com_ang_vel_b * 0.25,  # 3
                self._robot.data.projected_gravity_b,  # 3
                self._high_actions,  # 3 (velocity command)
                self._filter_actions,  # 3 (previous filter action)
            ],
            dim=-1,
        )
        self._filter_actions = self._filter_policy(filter_proprio_obs, self._obs_actor_rays)

        self._prev_loco_actions.append(self._loco_actions.clone())
        self._prev_loco_actions.pop(0)
        loco_cmd_scale = torch.tensor([[2.0, 2.0, 0.25]], device=self.device)
        loco_obs = torch.cat(
            [
                self._robot.data.root_com_ang_vel_b * 0.25,  # 3
                self._robot.data.projected_gravity_b,  # 3
                self._filter_actions.clamp(min=-self._vel_cmd_limits, max=self._vel_cmd_limits) * loco_cmd_scale,  # 3
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,  # 12
                self._robot.data.joint_vel * 0.05,  # 12
                self._loco_actions,  # 12
            ],
            dim=-1,
        )
        self._loco_actions = self._loco_policy(loco_obs)

        self._update_dynamic_obstacles()

    def _apply_action(self):
        actions = self._loco_actions * 0.8 + self._prev_loco_actions[-1] * 0.2
        action_scaled = actions * self.cfg.action_scale_loco
        action_scaled[:, self._hip_ids_jt] *= 0.5
        joint_pos_target = action_scaled + self._robot.data.default_joint_pos
        self._robot.set_joint_position_target(joint_pos_target)

    def _get_observations(self) -> dict:
        contact_indicators = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_cs, :], dim=-1) > 0.1
        contact_indicators = contact_indicators.float()

        dist_to_goal = torch.norm(self._cmd_goal_b[:, :2], dim=-1)
        goal_obs = self._cmd_goal_b[:, :2] / (dist_to_goal.unsqueeze(-1) + 1e-6)
        goal_obs *= dist_to_goal.unsqueeze(-1).clamp(max=3.0)

        obs_buf = torch.cat(
            [
                # proprioception
                self._robot.data.root_com_ang_vel_b * 0.25,  # 3
                self._robot.data.projected_gravity_b,  # 3
                self._cmd_goal_b[:, :2],  # 2
                # goal_obs,  # 2
                self._high_actions,  # 3
                # privileged information
                self._robot.data.root_com_lin_vel_b,  # 3
                self._guiding_vectors,  # 2
            ],
            dim=-1,
        )

        noise_buf = torch.cat(
            [
                torch.ones(3) * 0.2,  # angular velocity
                torch.ones(3) * 0.05,  # projected gravity
                torch.zeros(2),  # goal
                torch.zeros(self.cfg.num_high_actions),
                # privileged information
                torch.zeros(3),
                torch.zeros(2),
            ],
            dim=0,
        )
        obs_buf += (torch.rand_like(obs_buf) * 2.0 - 1.0) * noise_buf.to(self.device)

        obs_buf = self._augment_ray_obs(obs_buf)
        self._obs_buf[:] = obs_buf

        self._collect_data_extra()

        return {"policy": obs_buf}

    def _get_rewards(self) -> torch.Tensor:
        min_ttc = self._ttc_full.min(dim=-1).values
        vel_dir_b = self._robot.data.root_lin_vel_b[:, :2] / (
            torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=-1, keepdim=True) + 1e-6
        )
        robot_speed = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=-1)
        dist_to_goal = torch.norm(self._cmd_goal_w[:, :2] - self._robot.data.root_pos_w[:, :2], dim=-1)
        time_left = self._episode_length_sec_per_env - self.episode_length_buf * self.step_dt

        r_close_to_goal_1 = (dist_to_goal < 0.5).float() * 40.0
        r_close_to_goal_2 = (dist_to_goal < 1.0).float() * 15.0
        r_close_to_goal_3 = (dist_to_goal < 2.0).float() * 5.0
        bonus_ratio = torch.square(time_left / self._episode_length_sec_per_env)
        r_close_to_goal_1 *= bonus_ratio
        r_close_to_goal_2 *= bonus_ratio
        r_close_to_goal_3 *= bonus_ratio

        r_stop_at_goal = torch.square(robot_speed) * (dist_to_goal < 2.0) * -2.0
        robot_twist_norm = torch.norm(
            torch.cat([self._robot.data.root_lin_vel_b[:, :2], self._robot.data.root_ang_vel_b[:, 2:3]], dim=-1),
            dim=-1,
        )
        r_stay_at_goal = (
            torch.exp(-4.0 * robot_twist_norm) * (dist_to_goal < 2.0).float() * torch.exp(-2.0 * dist_to_goal) * 5.0
        )

        r_forward_vel = (vel_dir_b[:, 0] > 0.2).float() * (dist_to_goal >= 2.0).float() * 0.5
        r_backward_vel = (vel_dir_b[:, 0] < -0.2).float() * (dist_to_goal >= 2.0).float() * -4.0

        improved_dist = torch.clip(self._last_dist_to_goal - dist_to_goal, min=0.0)
        r_progress = improved_dist * 20.0

        guiding_mask = torch.norm(self._guiding_vectors, dim=-1) > 0.1
        guiding_dir = self._guiding_vectors / (torch.norm(self._guiding_vectors, dim=-1, keepdim=True) + 1e-6)
        robot_vel_w = self._robot.data.root_lin_vel_w[:, :2]
        robot_vel_dir = robot_vel_w / (torch.norm(robot_vel_w, dim=-1, keepdim=True) + 1e-6)
        align_score = torch.sum(guiding_dir * robot_vel_dir, dim=-1)
        r_follow_guide = (align_score > 0.7) * (dist_to_goal >= 2.0).float() * 1.0
        r_follow_guide[dist_to_goal < 2.0] = 1.0
        r_follow_guide[~guiding_mask] = 1.0
        r_not_follow_guide = (align_score < 0.1) * (dist_to_goal >= 2.0).float() * -5.0
        r_not_follow_guide[~guiding_mask] = 0.0

        action_diff = torch.sum(torch.square(self._high_actions - self._filter_actions), dim=-1)
        r_bad_control = (1.0 - torch.exp(-2.0 * action_diff)) * -0.5

        r_time_spent = torch.exp(-1.0 * time_left) * (dist_to_goal >= 2.0).float() * -5.0

        r_action_rate = torch.sum(torch.square(self._high_actions - self._prev_high_actions[-1]), dim=1)
        r_action_rate *= -0.01

        r_action_smoothness = torch.sum(
            torch.square(self._high_actions - 2.0 * self._prev_high_actions[-1] + self._prev_high_actions[-2]),
            dim=1,
        )
        r_action_smoothness *= -0.01

        r_cmd_limits = (torch.abs(self._high_actions[:, 0]) - self._vel_cmd_limits[0, 0]).clip(min=0.0).square()
        r_cmd_limits += (torch.abs(self._high_actions[:, 1]) - self._vel_cmd_limits[0, 1]).clip(min=0.0).square()
        r_cmd_limits += (torch.abs(self._high_actions[:, 2]) - self._vel_cmd_limits[0, 2]).clip(min=0.0).square()
        r_cmd_limits *= -10.0

        net_contact_forces = self._contact_sensor.data.net_forces_w
        r_collision = torch.sum(
            torch.norm(net_contact_forces[:, self._undesired_contact_body_ids_cs, :], dim=-1) > 0.1, dim=1
        ).float()
        r_collision *= -100.0

        r_termination = self._reset_buf.float() * (dist_to_goal > 2.0).float() * -100.0

        ray_distances = torch.norm(
            self._raycaster_measure.data.ray_hits_w - self._raycaster_measure.data.pos_w.unsqueeze(1), dim=-1
        )
        ray_distances.nan_to_num_(posinf=6.0).clip_(min=0.0, max=6.0)
        ray_distances = ray_distances.reshape(self.num_envs, self.cfg.num_ray_centers * 3, 180).min(dim=1).values
        dyn_obst_nearby = torch.any((self._ray_class > 0) & (ray_distances < 3.0), dim=-1)

        r_ttc = -0.5 * torch.exp(-2.0 * min_ttc)

        vel_dir_score = torch.sum(
            self._robot.data.root_lin_vel_b[:, None, :2] * self._ray_directions_b[:, :, :2], dim=-1
        )
        vel_dir_weight = torch.softmax(vel_dir_score / 1e-4, dim=-1)
        vel_ttc = torch.sum(vel_dir_weight * self._ttc_full, dim=-1)
        r_bad_vel = -2.0 * torch.exp(-2.0 * vel_ttc)

        self._last_dist_to_goal[:] = dist_to_goal[:]

        rewards = {k: v * self.step_dt for k, v in locals().items() if k.startswith("r_")}

        for k, v in rewards.items():
            if k not in self._episode_sums:
                self._episode_sums[k] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            if v.shape != self._episode_sums[k].shape:
                raise ValueError(f"reward {k} has wrong shape: {v.shape}, expected: {self._episode_sums[k].shape}")
            self._episode_sums[k] += v

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf * self.cfg.sim.dt * self.cfg.decimation >= self._episode_length_sec_per_env
        self._timeout_buf[:] = time_out

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id_cs], dim=-1), dim=1)[0] > 1.0,
            dim=1,
        )
        died |= -self._robot.data.projected_gravity_b[:, 2] < 0.75

        net_contact_forces = self._contact_sensor.data.net_forces_w
        collided = torch.any(
            torch.norm(net_contact_forces[:, self._undesired_contact_body_ids_cs, :], dim=-1) > 1.0, dim=1
        )
        died |= collided

        self._reset_buf[:] = died | time_out

        if torch.any(self._robot.data.root_pos_w[:, 2] < -5.0):
            print("\n\nROBOT IS FALLING DOWN!\n\n")

        if self.cfg.is_play_env and self._reset_tracked_env:
            died[self._track_env_id] = True
            self._reset_tracked_env = False

        return died, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        env_mask[env_ids] = True

        self._prev_grid_buf[env_mask] = 0.0
        self._prev_proj_gravity_buf[env_mask] = 0.0
        self._prev_ang_vel_buf[env_mask] = 0.0

        self._update_curriculum(env_ids)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._high_actions[env_ids] = 0.0
        self._filter_actions[env_ids] = 0.0
        self._loco_actions[env_ids] = 0.0
        for actions in self._prev_high_actions:
            actions[env_ids] = 0.0
        for actions in self._prev_loco_actions:
            actions[env_ids] = 0.0
        self._ttc_full[env_ids] = 3.0
        self._obs_buf[env_ids] = 0.0
        self._obs_actor_rays[env_ids] = 1.0

        self._reset_prev_extra_buf(env_ids)

        self._reset_robot_and_cmd(env_ids)

        self._last_dist_to_goal[env_ids] = torch.norm(
            self._cmd_goal_w[env_ids, :2] - self._robot.data.root_pos_w[env_ids, :2], dim=-1
        )

        self._reset_obstacles(env_ids)

        default_joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        default_joint_pos *= math_utils.sample_uniform(0.6, 1.4, default_joint_pos.shape, default_joint_pos.device)
        joint_pos_limits = self._robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = default_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        self._robot.write_joint_state_to_sim(joint_pos, self._robot.data.default_joint_vel[env_ids], env_ids=env_ids)

        self._loco_policy.hidden_state[:, env_mask, :] = 0.0
        self._loco_policy.cell_state[:, env_mask, :] = 0.0
        self._filter_policy.h_state[:, env_mask, :] = 0.0
        self._filter_policy.c_state[:, env_mask, :] = 0.0

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self._episode_length_sec
            self._episode_sums[key][env_ids] = 0.0
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Episode/average_speed"] = torch.mean(self._robot.data.root_com_lin_vel_b.norm(dim=1))
        extras["Episode/average_terrain_level"] = torch.mean(self._terrain.terrain_levels.to(torch.float))
        extras["Episode/max_terrain_level"] = torch.max(self._terrain.terrain_levels.to(torch.float))
        extras["Episode/mean_active_obstacles"] = torch.mean(self._num_active_obstacles.to(torch.float))
        extras["Episode/mean_max_episode_length"] = torch.mean(self._episode_length_sec_per_env)
        extras["Episode/env_type_1"] = torch.sum(self._env_type == 1).item()
        extras["Episode/env_type_2"] = torch.sum(self._env_type == 2).item()
        extras["Episode/env_type_3"] = torch.sum(self._env_type == 3).item()
        self.extras["log"] = extras

    def _reset_robot_and_cmd(self, env_ids: torch.Tensor):
        env_ids = env_ids.flatten()
        num_resets = env_ids.shape[0]
        if num_resets == 0:
            return

        env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        env_mask[env_ids] = True
        env_mask_1 = env_mask & (self._env_type == 1)
        env_mask_2 = env_mask & (self._env_type == 2)
        env_mask_3 = env_mask & (self._env_type == 3)
        num_resets_1 = torch.count_nonzero(env_mask_1).item()
        num_resets_2 = torch.count_nonzero(env_mask_2).item()
        num_resets_3 = torch.count_nonzero(env_mask_3).item()
        rand_patches_1 = torch.randint(0, self._flat_patches_1.shape[0], (num_resets_1,), device=self.device)
        rand_patches_2 = torch.randint(0, self._flat_patches_2.shape[0], (num_resets_2,), device=self.device)
        rand_patches_3 = torch.randint(0, self._flat_patches_3.shape[0], (num_resets_3,), device=self.device)

        self._p_n(
            f"total resets: {num_resets}"
            f", env type 1: {num_resets_1}"
            f", env type 2: {num_resets_2}"
            f", env type 3: {num_resets_3}",
            "num reset per env type",
            1,
        )

        root_state = self._robot.data.default_root_state.clone()
        zeros = torch.zeros(root_state.shape[0], device=self.device)
        rand_yaw = torch.rand_like(zeros) * math.pi * 2.0 - math.pi
        rand_quats = math_utils.quat_from_euler_xyz(zeros, zeros, rand_yaw)
        root_state[:, 3:7] = math_utils.quat_mul(root_state[:, 3:7], rand_quats)

        self._goal_idx_general[env_mask_1] = rand_patches_1
        self._robot_start_pos[env_mask_1, :] = self._flat_patches_1[rand_patches_1, :].clone()
        root_state[env_mask_1, :2] = self._robot_start_pos[env_mask_1, :2]
        self._cmd_goal_w[env_mask_1, :2] = self._flat_patches_1_goals[rand_patches_1, :2].clone()
        self._cmd_goal_w[env_mask_1, :2] += math_utils.sample_uniform(
            -0.25, 0.25, (num_resets_1, 2), device=self.device
        )

        self._goal_idx_dead_end[env_mask_2] = rand_patches_2
        self._robot_start_pos[env_mask_2, :] = self._flat_patches_2[rand_patches_2, :].clone()
        root_state[env_mask_2, :2] = self._robot_start_pos[env_mask_2, :2]
        self._cmd_goal_w[env_mask_2, :2] = self._flat_patches_2_goals[rand_patches_2, :2].clone()
        self._cmd_goal_w[env_mask_2, :2] += math_utils.sample_uniform(
            -0.25, 0.25, (num_resets_2, 2), device=self.device
        )

        self._goal_idx_detour[env_mask_3] = rand_patches_3
        self._robot_start_pos[env_mask_3, :] = self._flat_patches_3[rand_patches_3, :].clone()
        root_state[env_mask_3, :2] = self._robot_start_pos[env_mask_3, :2]
        self._cmd_goal_w[env_mask_3, :2] = self._flat_patches_3_goals[rand_patches_3, :2].clone()

        self._robot.write_root_link_pose_to_sim(root_state[env_ids, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(root_state[env_ids, 7:], env_ids)

        self._cmd_goal_b = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self._robot.data.root_quat_w), self._cmd_goal_w - self._robot.data.root_pos_w
        )
        cmd_goal_dir_norm = torch.norm(self._cmd_goal_b, dim=-1)
        self._cmd_goal_dir_b = self._cmd_goal_b / (cmd_goal_dir_norm[:, None] + 1e-6)
        self._cmd_goal_dir_b[cmd_goal_dir_norm < 0.1] = 0.0
        self._cmd_goal_heading_b = torch.atan2(self._cmd_goal_dir_b[:, 1], self._cmd_goal_dir_b[:, 0]).unsqueeze(-1)

    def _reset_obstacles(self, env_ids: torch.Tensor):
        disable_obstacle = math_utils.sample_uniform(0.0, 1.0, self.num_envs, device=self.device) < 0.1
        disable_obstacle = disable_obstacle[env_ids]
        self._no_obstacle_env[env_ids] = disable_obstacle

        robot_pos_xy = self._robot.data.root_pos_w[env_ids, :2]
        angle_deltas = torch.pi * 2.0 / (self._num_active_obstacles[env_ids].float().unsqueeze(-1) + 1e-6)
        angle_deltas.clip_(max=torch.pi)
        for i, obj in enumerate(self._obstacles):
            root_state_obj = obj.data.default_root_state[env_ids].clone()
            root_state_obj[:, 2] = math_utils.sample_uniform(-0.25, 0.75, root_state_obj.shape[0], device=self.device)
            rand_angles = angle_deltas * (i + 0.5)
            rand_dists = math_utils.sample_uniform(
                self._obst_dist_range[env_ids, 0:1],
                self._obst_dist_range[env_ids, 1:2],
                (len(env_ids), 1),
                device=self.device,
            )
            rand_offsets = torch.cat([torch.cos(rand_angles), torch.sin(rand_angles)], dim=-1) * rand_dists
            self._obst_pos_xy_a[env_ids, i, :] = robot_pos_xy + rand_offsets
            mask = math_utils.sample_uniform(0.0, 1.0, len(env_ids), device=self.device) < 0.1
            rand_angles_2 = rand_angles + angle_deltas * 2.0
            rand_offsets = torch.where(
                mask[:, None].expand(-1, 2),
                -rand_offsets,
                torch.cat([torch.cos(rand_angles_2), torch.sin(rand_angles_2)], dim=-1) * rand_dists,
            )
            self._obst_pos_xy_b[env_ids, i, :] = robot_pos_xy + rand_offsets
            root_state_obj[:, :2] = self._obst_pos_xy_a[env_ids, i, :]

            self._obst_speed[env_ids, i, :] = math_utils.sample_uniform(
                *self._obst_speed_range, (len(env_ids), 1), device=self.device
            )

            zeros = torch.zeros(root_state_obj.shape[0], device=self.device)
            rand_yaw = math_utils.sample_uniform(-math.pi, math.pi, (root_state_obj.shape[0],), device=self.device)
            root_state_obj[:, 3:7] = math_utils.quat_from_euler_xyz(zeros, zeros, rand_yaw)

            away_mask = i >= self._num_active_obstacles
            away_mask |= self._env_type == 2
            away_mask |= self._env_type == 3
            root_state_obj[away_mask[env_ids], 2] = -20.0

            root_state_obj[disable_obstacle, 2] = -20.0
            if self.cfg.no_obstacle:
                self._p_n("no obstacle activated!", "no obstacle", 100)
                root_state_obj[:, 2] = -20.0

            obj.write_root_state_to_sim(root_state_obj, env_ids)

    def _resolve_xy_velocity_to_arrow(
        self,
        xy_velocity: torch.Tensor,
        env_ids=None,
        world_frame: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        default_scale = self._goal_vel_viz.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        if not world_frame:
            base_quat_w = self._robot.data.root_quat_w[env_ids]
            arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _visualize_preprocessed_grid(self):
        if not self._show_debug_viz or self._current_grid is None:
            return

        env_id = 0
        grid = self._current_grid[env_id]  # Shape: (30, 180)

        phi_range = (-180, 180)
        theta_range = (-5, 55)
        phi_res_deg = 2
        theta_res_deg = 2

        theta_bins, phi_bins = grid.shape

        phi_indices = torch.arange(phi_bins, device=self.device)
        theta_indices = torch.arange(theta_bins, device=self.device)

        phi_grid, theta_grid = torch.meshgrid(phi_indices, theta_indices, indexing="ij")

        phi_angles = (phi_grid.float() + 0.5) * phi_res_deg + phi_range[0]
        theta_angles = (theta_grid.float() + 0.5) * theta_res_deg + theta_range[0]

        phi_rad = torch.deg2rad(phi_angles)
        theta_rad = torch.deg2rad(theta_angles)

        distances = grid.T  # Transpose to match phi, theta indexing

        valid_mask = distances < self._proximal_ray_dist * 1.05

        if valid_mask.sum() == 0:
            return

        x = distances * torch.cos(theta_rad) * torch.cos(phi_rad)
        y = distances * torch.cos(theta_rad) * torch.sin(phi_rad)
        z = distances * torch.sin(theta_rad)

        valid_x = x[valid_mask]
        valid_y = y[valid_mask]
        valid_z = z[valid_mask]
        valid_distances = distances[valid_mask]

        robot_pos = self._robot.data.root_pos_w[env_id]
        robot_quat = self._robot.data.root_quat_w[env_id]

        points_robot = torch.stack([valid_x, valid_y, valid_z], dim=-1)
        points_world = math_utils.quat_apply(robot_quat.unsqueeze(0), points_robot) + robot_pos

        normalized_dist = (valid_distances / self._proximal_ray_dist).clamp(0, 1)
        colors = []
        for dist in normalized_dist:
            r = 1.0 - dist.item()
            g = 0.6
            b = dist.item()
            colors.append((r, g, b, 0.8))

        points_list = [(p[0].item(), p[1].item(), p[2].item()) for p in points_world]
        sizes = [10] * len(points_list)

        if len(points_list) > 0:
            self._debug_draw.draw_points(points_list, colors, sizes)

    def _init_physx_material_buffer(self):
        self._num_shapes_per_body = []
        for link_path in self._robot.root_physx_view.link_paths[0]:
            link_physx_view = self._robot._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
            self._num_shapes_per_body.append(link_physx_view.max_shapes)
        num_shapes = sum(self._num_shapes_per_body)
        expected_shapes = self._robot.root_physx_view.max_shapes
        if num_shapes != expected_shapes:
            raise ValueError(
                "Failed to parse the number of shapes per body."
                f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
            )

        self._num_physx_mat_buckets = 64
        range_list = [self.cfg.static_friction_range, self.cfg.dynamic_friction_range, self.cfg.restitution_range]
        ranges = torch.tensor(range_list, device="cpu")
        self._material_buckets = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (self._num_physx_mat_buckets, 3), device="cpu"
        )

        self._physics_parameters = torch.zeros(self.num_envs, 3, device=self.device)

    def _reset_physx_materials(self, env_ids):
        bucket_ids = torch.randint(0, self._num_physx_mat_buckets, (len(env_ids),), device="cpu")
        material_samples = self._material_buckets[bucket_ids]
        self._physics_parameters[env_ids, :] = material_samples.to(self.device)

        materials = self._robot.root_physx_view.get_material_properties()

        materials[env_ids, :] = material_samples.reshape(-1, 1, 3)

        self._robot.root_physx_view.set_material_properties(materials, env_ids)

    def _randomize_mass(self):
        asset = self._robot

        env_ids = torch.arange(self.num_envs, device="cpu")

        body_ids, _ = self._robot.find_bodies("base")
        body_ids = torch.tensor(body_ids, dtype=torch.int, device="cpu")
        assert body_ids.shape == (1,), "Mass randomization is only supported for the base body."

        masses = asset.root_physx_view.get_masses()
        print("*" * 50)
        print(f"masses before randomization: {masses[env_ids, body_ids]}")
        print("*" * 50)

        masses[env_ids[:, None], body_ids] = asset.data.default_mass[env_ids[:, None], body_ids].clone()

        masses[env_ids[:, None], body_ids] += math_utils.sample_uniform(
            -1.0, 2.0, (masses.shape[0], body_ids.shape[0]), device=masses.device
        )

        asset.root_physx_view.set_masses(masses, env_ids)

        ratios = masses[env_ids[:, None], body_ids] / asset.data.default_mass[env_ids[:, None], body_ids]
        inertias = asset.root_physx_view.get_inertias()
        if isinstance(asset, Articulation):
            inertias[env_ids[:, None], body_ids] = (
                asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
            )
        else:
            inertias[env_ids] = asset.data.default_inertia[env_ids] * ratios
        asset.root_physx_view.set_inertias(inertias, env_ids)

        new_masses = asset.root_physx_view.get_masses()
        print("*" * 50)
        print(f"randomized masses: {new_masses[env_ids, body_ids]}")
        print("*" * 50)

    def _augment_ray_obs(self, obs_buf):
        self._ray_directions_b = self._raycaster_measure.ray_directions[:, 0:180, :].clone()

        ray_dists = torch.norm(
            self._raycaster_measure.data.ray_hits_w[:, :, :] - self._robot.data.root_pos_w[:, None, :], dim=-1
        )
        ray_dists.nan_to_num_(nan=6.0, posinf=6.0, neginf=6.0).clip_(min=0.0, max=6.0)

        gt_rays = (
            ray_dists.reshape(self.num_envs, self.cfg.num_ray_centers * 3, 180)[:, :, :].min(dim=1).values.clip(max=3.0)
            / 3.0
        )

        actor_rays = ray_dists[:, : 180 * 3].reshape(self.num_envs, 3, 180).min(dim=1).values.clip(max=3.0) / 3.0
        actor_rays += math_utils.sample_uniform(-0.05, 0.05, actor_rays.shape, device=self.device)
        assert actor_rays.shape[1] == self.cfg.obs_dims["actor_ray"]

        if self.cfg.is_play_env:
            actor_rays[:, :] = gt_rays[:, :]

        critic_rays = ray_dists[:, : 180 * 3].clone() / 6.0
        assert critic_rays.shape[1] == self.cfg.obs_dims["critic_ray"]

        closest_ray_idx = ray_dists.reshape(self.num_envs, self.cfg.num_ray_centers * 3, 180).argmin(dim=1)
        if self._raycaster_measure._ray_mesh_ids is not None:
            ray_hit_mesh_ids = self._raycaster_measure._ray_mesh_ids.reshape(
                self.num_envs, self.cfg.num_ray_centers * 3, 180
            )[
                torch.arange(self.num_envs, device=self.device).unsqueeze(-1),
                closest_ray_idx,
                torch.arange(180, device=self.device),
            ]
            self._ray_hit_mesh_ids = ray_hit_mesh_ids.clone()
            self._viz_ray_hit_mesh_ids = ray_hit_mesh_ids[0].cpu().numpy()

        if self._data_writer is not None or (self._ray_predictor is not None and self.cfg.use_predicted_rays):
            grid_data = self.preprocess_lidar_frame(device=self.device)
            self._current_grid = grid_data
            self._prev_grid_buf = torch.cat([self._prev_grid_buf[:, 1:, :, :], grid_data[:, None, :, :]], dim=1)

            gravity = self._robot.data.projected_gravity_b
            gravity += math_utils.sample_uniform(-0.1, 0.1, gravity.shape, device=self.device)
            self._prev_proj_gravity_buf = torch.cat([self._prev_proj_gravity_buf[:, 1:, :], gravity[:, None, :]], dim=1)

            ang_vel = self._robot.data.root_ang_vel_b
            ang_vel += math_utils.sample_uniform(-0.1, 0.1, ang_vel.shape, device=self.device)
            self._prev_ang_vel_buf = torch.cat([self._prev_ang_vel_buf[:, 1:, :], ang_vel[:, None, :]], dim=1)

            if self._data_writer is not None:
                self._data_writer.add_data(
                    self.num_envs,
                    self._prev_grid_buf.cpu().numpy(),
                    self._prev_proj_gravity_buf.cpu().numpy(),
                    self._prev_ang_vel_buf.cpu().numpy(),
                    gt_rays.cpu().numpy(),
                )

            if self._ray_predictor is not None and self.cfg.use_predicted_rays:
                with torch.no_grad():
                    grid_data = self._prev_grid_buf / self._proximal_ray_dist
                    gravity_vectors = self._prev_proj_gravity_buf
                    ang_vel = self._prev_ang_vel_buf
                    pred_actor_rays = self._ray_predictor(grid_data, torch.cat([gravity_vectors, ang_vel], dim=-1))
                actor_rays[:, :] = pred_actor_rays[:, :]
                self._p_n("actor rays modified!", "pred rays", 10)

        if self._show_debug_viz is True:
            self._viz_ray_goodness = (
                self._raycaster_measure.data.pos_w[self._track_env_id].unsqueeze(0)  #
                + math_utils.quat_apply_yaw(
                    self._robot.data.root_quat_w[self._track_env_id, None, :],
                    self._ray_directions_b[self._track_env_id],
                )  #
                * actor_rays[self._track_env_id].unsqueeze(-1)
                * 1.0
            )

        robot_pos_general = self._robot.data.root_pos_w[self._env_type == 1, :]
        robot_pos_dead_end = self._robot.data.root_pos_w[self._env_type == 2, :]
        robot_pos_detour = self._robot.data.root_pos_w[self._env_type == 3, :]
        guiding_vectors_general = self._general_env_guide.get_optimal_direction(
            x=robot_pos_general[:, 0] + self.border_x,
            y=robot_pos_general[:, 1] + self.border_y,
            goal_index=self._goal_idx_general[self._env_type == 1],
        )
        guiding_vectors_dead_end = self._dead_end_env_guide.get_optimal_direction(
            x=robot_pos_dead_end[:, 0] + self.border_x,
            y=robot_pos_dead_end[:, 1] + self.border_y,
            goal_index=self._goal_idx_dead_end[self._env_type == 2],
        )
        guiding_vectors_detour = self._detour_env_guide.get_optimal_direction(
            x=robot_pos_detour[:, 0] + self.border_x,
            y=robot_pos_detour[:, 1] + self.border_y,
            goal_index=self._goal_idx_detour[self._env_type == 3],
        )
        self._guiding_vectors[self._env_type == 1, :] = guiding_vectors_general
        self._guiding_vectors[self._env_type == 2, :] = guiding_vectors_dead_end
        self._guiding_vectors[self._env_type == 3, :] = guiding_vectors_detour

        self._obs_actor_rays = actor_rays.clone()
        obs_buf = torch.cat(
            [obs_buf, actor_rays, critic_rays],
            dim=1,
        )

        self._compute_ttc()

        return obs_buf

    def _compute_ttc(self):
        ray_distances = torch.norm(
            self._raycaster_measure.data.ray_hits_w - self._raycaster_measure.data.pos_w.unsqueeze(1), dim=-1
        )
        ray_distances.nan_to_num_(posinf=6.0).clip_(min=0.0, max=6.0)
        ray_distances = ray_distances.reshape(self.num_envs, self.cfg.num_ray_centers * 3, 180).min(dim=1).values

        ttc_full = torch.ones(self.num_envs, 180, device=self.device) * 3.0
        robot_lin_vel_b = self._robot.data.root_lin_vel_b[:, :2]
        ray_dir_b_2d = self._ray_directions_b[:, :, :2]
        proj_vel = torch.sum(robot_lin_vel_b[:, None, :] * ray_dir_b_2d, dim=-1).clip(min=0.0)
        for i in range(1, self._num_obstacles + 1):
            mask = self._ray_hit_mesh_ids == i
            if torch.sum(mask) < 1:
                continue
            obj_lin_vel_w = self._obst_pos_xy_b[:, i - 1] - self._obst_pos_xy_a[:, i - 1]
            obj_lin_vel_w /= torch.norm(obj_lin_vel_w, dim=-1, keepdim=True) + 1e-6
            obj_lin_vel_w *= self._obst_speed[:, i - 1, :]

            ray_dir_w_2d = self._ray_directions_b[:, :, :]
            ray_dir_w_2d = math_utils.quat_apply_yaw(
                self._robot.data.root_quat_w[:, None, :].expand(-1, ray_dir_b_2d.shape[1], -1).contiguous(),
                ray_dir_w_2d,
            )[:, :, :2]  # (N, 180, 2)
            obj_proj_vel = torch.sum(obj_lin_vel_w[:, None, :] * ray_dir_w_2d, dim=-1)

            rel_vel = (proj_vel - obj_proj_vel).clip(min=0.0)
            dynamic_ttc = (ray_distances.clip(max=3.0) / (rel_vel + 1e-6)).clip(max=3.0)
            ttc_full[mask] = dynamic_ttc[mask]

        static_ttc = (ray_distances.clip(max=3.0) / (proj_vel + 1e-6)).clip(max=3.0)
        mask = self._ray_hit_mesh_ids == 0
        ttc_full[mask] = static_ttc[mask]

        self._ttc_full[:, :] = ttc_full.clone()
        self._ray_class = (self._ray_hit_mesh_ids > 0).float()

    def preprocess_lidar_frame(
        self,
        phi_range: tuple[float, float] = (-180, 180),
        theta_range: tuple[float, float] = (-5, 55),
        phi_res_deg: int = 2,
        theta_res_deg: int = 2,
        device: str = "cuda",
    ):
        rc_offset = torch.zeros(self.num_envs, 3, device=self.device)
        rc_offset[:, 0] = self.cfg.raycaster.offset.pos[0]
        rc_offset[:, 1] = self.cfg.raycaster.offset.pos[1]
        rc_offset[:, 2] = self.cfg.raycaster.offset.pos[2]
        rc_pos_w = self._raycaster.data.pos_w + math_utils.quat_apply(self._raycaster.data.quat_w, rc_offset)

        ray_directions = self._raycaster.data.ray_hits_w - rc_pos_w.unsqueeze(1)
        ray_distances = torch.norm(ray_directions, dim=-1)
        ray_distances.nan_to_num_(posinf=10.0).clip_(min=0.0, max=10.0)

        small_noise = math_utils.sample_uniform(-0.1, 0.1, ray_distances.shape, device=self.device)
        ray_distances += small_noise

        ray_hits_b = rc_offset.unsqueeze(1) + self._raycaster.ray_directions * ray_distances.unsqueeze(-1)
        assert torch.all(torch.isfinite(ray_hits_b))

        batch_size, num_points = ray_hits_b.shape[0], ray_hits_b.shape[1]

        phi_res = torch.deg2rad(torch.tensor(phi_res_deg, device=device))
        theta_res = torch.deg2rad(torch.tensor(theta_res_deg, device=device))

        phi_bins = int((phi_range[1] - phi_range[0]) / phi_res_deg)
        theta_bins = int((theta_range[1] - theta_range[0]) / theta_res_deg)

        points = ray_hits_b.reshape(-1, num_points, 3)  # (B*H, N, 3)

        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        r = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.asin(z / (r + 1e-8))  # Elevation
        phi = torch.atan2(y, x)  # Azimuth

        phi_min, phi_max = torch.deg2rad(torch.tensor(phi_range, device=device))
        theta_min, theta_max = torch.deg2rad(torch.tensor(theta_range, device=device))

        valid_mask = (phi >= phi_min) & (phi <= phi_max) & (theta >= theta_min) & (theta <= theta_max)
        valid_mask &= r < self._proximal_ray_dist

        phi_indices = torch.floor((phi - phi_min) / phi_res).long()
        theta_indices = torch.floor((theta - theta_min) / theta_res).long()

        phi_indices = torch.clamp(phi_indices, 0, phi_bins - 1)
        theta_indices = torch.clamp(theta_indices, 0, theta_bins - 1)

        grid_shape = (points.shape[0], theta_bins, phi_bins)
        grid = torch.full(grid_shape, self._proximal_ray_dist, device=device, dtype=torch.float32)

        batch_indices = torch.arange(points.shape[0], device=device).unsqueeze(1).expand_as(r)
        flat_indices = batch_indices * (phi_bins * theta_bins) + theta_indices * phi_bins + phi_indices

        grid_flat = grid.reshape(-1)
        grid_flat.scatter_reduce_(0, flat_indices[valid_mask], r[valid_mask], reduce="amin", include_self=False)
        grid = grid_flat.reshape(grid_shape)

        return grid

    def _p_n(self, msg: str, id: str, n: int = 10):
        """Print a message up to n times."""
        if not hasattr(self, "_print_counter"):
            self._print_counter = {}
        if id not in self._print_counter:
            self._print_counter[id] = 0
        if self._print_counter[id] < n:
            print(msg)
            self._print_counter[id] += 1

    def _p_by_n(self, msg: str, id: str, n: int = 100):
        """Print a message every n times."""
        if not hasattr(self, "_print_counter_2"):
            self._print_counter_2 = {}
        if id not in self._print_counter_2:
            self._print_counter_2[id] = 0
        if self._print_counter_2[id] % n == 0:
            print(msg)
        self._print_counter_2[id] += 1

    async def setup_ui(self):
        if not self.cfg.is_play_env or self.viewport_camera_controller is None:
            return

        import omni
        import omni.ui as ui

        def next_tracked_env():
            self._track_env_id += 1
            if self._track_env_id >= self.num_envs:
                self._track_env_id = 0
            print(f"Now tracking env {self._track_env_id}")

        def prev_tracked_env():
            self._track_env_id -= 1
            if self._track_env_id < 0:
                self._track_env_id = self.num_envs - 1
            print(f"Now tracking env {self._track_env_id}")

        def inc_tracked_curriculum():
            self._track_env_curriculum += 1
            if self._track_env_curriculum >= self.num_terrain_rows:
                self._track_env_curriculum = self.num_terrain_rows - 1
            print(f"Now tracking curriculum {self._track_env_curriculum}")

        def dec_tracked_curriculum():
            self._track_env_curriculum -= 1
            if self._track_env_curriculum < 0:
                self._track_env_curriculum = 0
            print(f"Now tracking curriculum {self._track_env_curriculum}")

        def inc_camera_height():
            self._track_camera_height += 1.0
            if self._track_camera_height > 30.0:
                self._track_camera_height = 30.0
            print(f"Now tracking camera height {self._track_camera_height}")

        def dec_camera_height():
            self._track_camera_height -= 1.0
            if self._track_camera_height < 1.0:
                self._track_camera_height = 1.0
            print(f"Now tracking camera height {self._track_camera_height}")

        def reset_tracked_env():
            if self._track_env_id >= 0 and self._track_env_id < self.num_envs:
                self._reset_tracked_env = True
                print(f"Reset env {self._track_env_id}")

        self._control_window = ui.Window("Simulation Controls", noTabBar=True)
        await omni.kit.app.get_app().next_update_async()
        stage_window = ui.Workspace.get_window("Stage")
        self._control_window.dock_in(stage_window, ui.DockPosition.SAME)
        sim_settings_window = ui.Workspace.get_window("Simulation Settings")
        if sim_settings_window is not None:
            sim_settings_window.dock_in(stage_window, ui.DockPosition.SAME)

        with self._control_window.frame:
            with ui.VStack(spacing=5, height=0):
                ui.Label("Switching Tracked Env", style={"color": 0xFFFFFFFF, "font_size": 16})
                ui.Button("Next Env", clicked_fn=next_tracked_env)
                ui.Button("Prev Env", clicked_fn=prev_tracked_env)
                ui.Spacer(height=10)
                ui.Label("Switching Tracked Curriculum", style={"color": 0xFFFFFFFF, "font_size": 16})
                ui.Button("Increase Curriculum", clicked_fn=inc_tracked_curriculum)
                ui.Button("Decrease Curriculum", clicked_fn=dec_tracked_curriculum)
                ui.Spacer(height=10)
                ui.Label("Adjust Camera Height", style={"color": 0xFFFFFFFF, "font_size": 16})
                ui.Button("Increase Height", clicked_fn=inc_camera_height)
                ui.Button("Decrease Height", clicked_fn=dec_camera_height)
                ui.Spacer(height=10)
                ui.Label("Resetting Tracked Env", style={"color": 0xFFFFFFFF, "font_size": 16})
                ui.Button("Reset Env", clicked_fn=reset_tracked_env)

    def _init_data_extra_buf(self):
        self._extra_num_hist_obs = 30
        self._extra_num_input_dim = 180 + 3
        self._extra_num_output_dim = 180 * 2
        self._prev_extra_buf = torch.zeros(
            self.num_envs,
            self._extra_num_hist_obs,
            self._extra_num_input_dim + self._extra_num_output_dim,
            device=self.device,
        )

    def _reset_prev_extra_buf(self, env_ids):
        self._prev_extra_buf[env_ids] = 0.0
        a, b = 0, 180
        self._prev_extra_buf[env_ids, a:b] = 1.0
        a, b = self._extra_num_input_dim, self._extra_num_input_dim + 180
        self._prev_extra_buf[env_ids, a:b] = 1.0

    def _collect_data_extra(self):
        if self._data_writer_extra is None:
            return

        ray_dists = torch.norm(
            self._raycaster_measure.data.ray_hits_w[:, :, :] - self._robot.data.root_pos_w[:, None, :], dim=-1
        )
        ray_dists.nan_to_num_(nan=6.0, posinf=6.0, neginf=6.0)
        ray_dists += math_utils.sample_uniform(-0.1, 0.1, ray_dists.shape, device=self.device)
        ray_dists.clip_(min=0.0, max=3.0)
        ray_dists = (
            ray_dists.reshape(self.num_envs, self.cfg.num_ray_centers * 3, 180).min(dim=1).values.clip(max=3.0) / 3.0
        )

        base_lin_vel_b = self._robot.data.root_lin_vel_b[:, :2]
        base_lin_vel_b += math_utils.sample_uniform(-0.2, 0.2, base_lin_vel_b.shape, device=self.device)
        base_ang_vel_b = self._robot.data.root_ang_vel_b[:, 2:3]
        base_ang_vel_b += math_utils.sample_uniform(-0.2, 0.2, base_ang_vel_b.shape, device=self.device)

        self._prev_extra_buf = torch.roll(self._prev_extra_buf, shifts=-1, dims=1)
        self._prev_extra_buf[:, -1, :] = torch.cat(
            [
                # inputs
                ray_dists,
                base_lin_vel_b,
                base_ang_vel_b,
                # outputs
                self._ttc_full[:, :] / 3.0,
                self._ray_class[:, :].float(),
            ],
            dim=-1,
        )

        self._data_writer_extra.add_data(
            num_envs=self.num_envs,
            input=self._prev_extra_buf[:, :, : self._extra_num_input_dim].cpu().numpy(),
            output=self._prev_extra_buf[:, -1, -self._extra_num_output_dim :].cpu().numpy(),
        )

    def _wait_for_key(self):
        if self.cfg.wait_for_key is False:
            return
        input("Press Enter to continue...")

    def _init_data_collection(self):
        self._data_writer = None
        self._data_writer_extra = None
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if self.cfg.data_collection_type == "train":
            self._data_writer = HDF5DatasetWriter_Ray(
                f"{script_dir}/../../ray_predictor/data/train.h5",
                500000,
                history_frames=self._num_prev_data,
                overwrite=True,
            )
            print("=" * 50)
            print("collecting training data...")
            print("=" * 50)
        elif self.cfg.data_collection_type == "val":
            self._data_writer = HDF5DatasetWriter_Ray(
                f"{script_dir}/../../ray_predictor/data/val.h5",
                50000,
                history_frames=self._num_prev_data,
                overwrite=True,
            )
            print("=" * 50)
            print("collecting validation data...")
            print("=" * 50)
        elif self.cfg.data_collection_type == "train_extra":
            self._data_writer_extra = HDF5DatasetWriter_General(
                f"{script_dir}/../../ray_predictor/data/train_extra.h5",
                input_dim=self._extra_num_input_dim,
                output_dim=self._extra_num_output_dim,
                num_total=500000,
                history_frames=self._extra_num_hist_obs,
                overwrite=True,
            )
            print("=" * 50)
            print("collecting training data: [extra]...")
            print("=" * 50)
        elif self.cfg.data_collection_type == "val_extra":
            self._data_writer_extra = HDF5DatasetWriter_General(
                f"{script_dir}/../../ray_predictor/data/val_extra.h5",
                input_dim=self._extra_num_input_dim,
                output_dim=self._extra_num_output_dim,
                num_total=50000,
                history_frames=self._extra_num_hist_obs,
                overwrite=True,
            )
            print("=" * 50)
            print("collecting validation data: [extra]...")
            print("=" * 50)
        else:
            print("=" * 50)
            print("NOT collecting data.")
            print("=" * 50)

    def _init_debug_viz(self):
        self._show_debug_viz = False
        if self.cfg.is_play_env and self.viewport_camera_controller is not None:
            print("\n********************************************")
            print("           **Enabling debug draw.**")
            print("********************************************\n")
            self._show_debug_viz = True
            # fmt: off
            import isaacsim
            from isaacsim.core.utils.extensions import enable_extension
            enable_extension("omni.isaac.debug_draw")
            from isaacsim.util.debug_draw import _debug_draw
            # fmt: on
            self._debug_draw = _debug_draw.acquire_debug_draw_interface()

            ray_viz_cfg = copy.deepcopy(self.cfg.raycaster.visualizer_cfg)
            ray_viz_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 0.3, 1.0)
            ray_viz_cfg.markers["hit"].radius = 0.02
            self._ray_viz = VisualizationMarkers(cfg=ray_viz_cfg)
            self._ray_viz.set_visibility(True)

            goal_viz_cfg = copy.deepcopy(self.cfg.raycaster.visualizer_cfg)
            goal_viz_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
            goal_viz_cfg.markers["hit"].radius = 0.2
            self._goal_viz = VisualizationMarkers(cfg=goal_viz_cfg)
            self._goal_viz.set_visibility(True)

            goal_vel_viz_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_goal")
            cur_vel_viz_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_current")
            action_vel_viz_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_safe")
            guide_field_viz_cfg = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_safe")
            goal_vel_viz_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
            cur_vel_viz_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
            action_vel_viz_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
            guide_field_viz_cfg.markers["arrow"].scale = (1.5, 1.5, 1.5)
            guide_field_viz_cfg.markers["arrow"].visual_material.diffuse_color = (0.0, 0.05, 0.0)

            self._goal_vel_viz = VisualizationMarkers(cfg=goal_vel_viz_cfg)
            self._cur_vel_viz = VisualizationMarkers(cfg=cur_vel_viz_cfg)
            self._action_vel_viz = VisualizationMarkers(cfg=action_vel_viz_cfg)
            self._guide_field_viz = VisualizationMarkers(cfg=guide_field_viz_cfg)
            self._goal_vel_viz.set_visibility(True)
            self._cur_vel_viz.set_visibility(True)
            self._action_vel_viz.set_visibility(True)
            self._guide_field_viz.set_visibility(True)

    def _create_obstacles(self):
        self._obstacles: list[RigidObject] = []
        self._num_active_obstacles = torch.ones(self.num_envs, dtype=torch.long, device=self.device) * 3
        self._obstacle_radius = 0.45
        obstacle_height = 1.5
        self._obstacle_prim_paths: list[str] = []
        cylinder_cfg = sim_utils.MeshCylinderCfg(
            radius=1.0,
            height=obstacle_height,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash_Planks.mdl",
                project_uvw=True,
                texture_scale=(1.0, 1.0),
                albedo_brightness=0.2,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, restitution=0.0),
        )
        box_cfg = sim_utils.MeshCuboidCfg(
            size=(1.0, 1.0, obstacle_height),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Ash_Planks.mdl",
                project_uvw=True,
                texture_scale=(1.0, 1.0),
                albedo_brightness=0.2,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, restitution=0.0),
        )
        assets_cfg = [
            cylinder_cfg.replace(radius=0.2),
            cylinder_cfg.replace(radius=0.3),
            box_cfg.replace(size=(0.4, 0.4, obstacle_height)),
            box_cfg.replace(size=(0.5, 0.5, obstacle_height)),
        ]
        for i in range(self._num_obstacles):
            prim_path = f"/World/envs/env_.*/Obstacle_{i}"
            self._obstacle_prim_paths.append(prim_path)
            obstacle_obj_cfg = RigidObjectCfg(
                prim_path=prim_path,
                spawn=sim_utils.MultiAssetSpawnerCfg(
                    assets_cfg=assets_cfg,
                    random_choice=True,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(),
            )
            obstacle_obj = RigidObject(cfg=obstacle_obj_cfg)
            self._obstacles.append(obstacle_obj)
            self.scene.rigid_objects[f"obstacle_{i}"] = obstacle_obj
            self.cfg.raycaster.mesh_prim_paths.append(prim_path)
            self.cfg.raycaster_measure.mesh_prim_paths.append(prim_path)
            assets_cfg = assets_cfg[-1:] + assets_cfg[:-1]  # rotate the assets for the next obstacle

    def _update_dynamic_obstacles(self):
        for i, obst in enumerate(self._obstacles):
            root_state = obst.data.root_state_w.clone()
            dist_to_pos_b = torch.norm(self._obst_pos_xy_b[:, i, :] - root_state[:, :2], dim=-1)
            update_mask = dist_to_pos_b < 0.5
            num_update = update_mask.sum().item()

            tmp = self._obst_pos_xy_a[update_mask, i, :].clone()
            self._obst_pos_xy_a[update_mask, i, :] = self._obst_pos_xy_b[update_mask, i, :].clone()
            self._obst_pos_xy_b[update_mask, i, :] = tmp

            self._obst_speed[update_mask, i, :] = math_utils.sample_uniform(
                *self._obst_speed_range, (num_update, 1), device=self.device
            )

            vel_dir = self._obst_pos_xy_b[:, i, :] - root_state[:, :2]
            vel_dir /= torch.norm(vel_dir, dim=-1, keepdim=True) + 1e-6
            root_state[:, :2] += vel_dir * self._obst_speed[:, i, :] * self.step_dt

            obst.write_root_state_to_sim(root_state)

    def _update_curriculum(self, env_ids):
        if not self._first_reset:
            dist_robot_to_goal = torch.norm(
                self._cmd_goal_w[env_ids, :2] - self._robot.data.root_pos_w[env_ids, :2],
                dim=-1,
            )
            dist_origin_to_goal = torch.norm(
                self._cmd_goal_w[env_ids, :2] - self._robot_start_pos[env_ids, :2],
                dim=-1,
            )
            move_up = dist_robot_to_goal < 2.0
            move_down = dist_robot_to_goal > dist_origin_to_goal * 0.5
            move_down *= ~move_up

            self._terrain.terrain_levels[env_ids] += 1 * move_up - 1 * move_down

            self._terrain.terrain_levels[env_ids] = torch.where(
                self._terrain.terrain_levels[env_ids] >= self.num_terrain_rows,
                torch.randint_like(self._terrain.terrain_levels[env_ids], self.num_terrain_rows),
                torch.clip(self._terrain.terrain_levels[env_ids], 0),
            )

            self._terrain.env_origins[env_ids, :] = self._terrain.terrain_origins[
                self._terrain.terrain_levels[env_ids], self._env_terrain_cols[env_ids], :
            ]
        else:
            self._first_reset = False

        self._num_active_obstacles[env_ids] = torch.clip(
            self._terrain.terrain_levels[env_ids] * 1.0, 0, self._num_obstacles
        ).long()

        if self.cfg.is_play_env:
            self._terrain.terrain_levels[self._track_env_id] = self._track_env_curriculum

        self._goal_dist_range[env_ids, 0] = (self._terrain.terrain_levels[env_ids].float() * 0.2) + 1.0
        self._goal_dist_range[env_ids, 1] = (self._terrain.terrain_levels[env_ids].float() * 0.5) + 3.0

        self._obst_dist_range[env_ids, 0] = 2.0
        self._obst_dist_range[env_ids, 1] = self._goal_dist_range[env_ids, 1]

    def _update_debug_draw(self):
        if not self._show_debug_viz:
            return

        self._debug_draw.clear_lines()
        self._debug_draw.clear_points()

        # self._visualize_preprocessed_grid()

        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._cmd_goal_dir_b[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self._high_actions[:, :2])
        # self._goal_vel_viz.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        # self._cur_vel_viz.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

        if hasattr(self, "_viz_ray_goodness"):
            mesh_palette = [
                (0.2, 0.7, 1.0, 1.0),  # terrain
                (1.0, 0.2, 0.2, 1.0),  # obstacle 1
                (0.2, 1.0, 0.2, 1.0),  # obstacle 2
                (1.0, 1.0, 0.2, 1.0),  # obstacle 3
                (1.0, 0.2, 1.0, 1.0),  # obstacle 4
                (0.2, 1.0, 1.0, 1.0),  # obstacle 5
                (0.2, 1.0, 1.0, 1.0),  # obstacle 5
                (0.2, 1.0, 1.0, 1.0),  # obstacle 5
                (0.2, 1.0, 1.0, 1.0),  # obstacle 5
                (0.2, 1.0, 1.0, 1.0),  # obstacle 5
                (0.0, 0.0, 0.0, 1.0),  # no hit color
            ]
            colors = []
            for mesh_id in self._viz_ray_hit_mesh_ids:
                if mesh_id < len(mesh_palette):
                    colors.append(mesh_palette[mesh_id])
                else:
                    raise ValueError(f"mesh_id {mesh_id} exceeds palette size {len(mesh_palette)}")

            self._debug_draw.draw_lines(
                [(p[0].item(), p[1].item(), p[2].item() - 0.2) for p in self._viz_ray_goodness],
                [
                    (
                        self._raycaster_measure.data.pos_w[self._track_env_id, 0].item(),
                        self._raycaster_measure.data.pos_w[self._track_env_id, 1].item(),
                        self._raycaster_measure.data.pos_w[self._track_env_id, 2].item() - 0.2,
                    )
                ]
                * self._viz_ray_goodness.shape[0],
                [(0.4, 0.7, 1.0, 0.6)] * self._viz_ray_goodness.shape[0],
                [3] * self._viz_ray_goodness.shape[0],
            )

            points = self._viz_ray_goodness + torch.tensor([[0.0, 0.0, -0.2]], device=self.device)
            self._ray_viz.visualize(points)

        self._goal_viz.visualize(self._cmd_goal_w)
