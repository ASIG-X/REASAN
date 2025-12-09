# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import asyncio
import math

import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import (
    Articulation,
)
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.sensors import (
    ContactSensor,
)
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from go2_lidar.tasks.go2_loco_env_cfg import Go2LocoEnvCfg


class Go2LocoEnv(DirectRLEnv):
    cfg: Go2LocoEnvCfg

    def __init__(self, cfg: Go2LocoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self.cfg.is_second_stage:
            print("second stage training.")
        else:
            print("first stage training.")

        if not self.cfg.is_second_stage:
            self._randomize_mass(-1.0, 2.0)
        else:
            self._randomize_mass(-2.0, 5.0)

        self._reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._init_physx_material_buffer()
        self._reset_physx_materials(torch.ones(self.num_envs, device="cpu", dtype=torch.bool))

        for _, v in self._robot.actuators.items():
            v.stiffness[:] = (torch.rand_like(v.stiffness) * 0.2 + 0.9) * 35.0
            v.damping[:] = (torch.rand_like(v.damping) * 0.2 + 0.9) * 0.5

        self._swing_peak = torch.zeros(self.num_envs, 4, device=self.device)

        self._first_reset = True
        self._actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self._previous_actions = [self._actions.clone()] * 5
        self._num_actions = self._actions.shape[1]

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

        self._cmd_lin_vel = torch.zeros(self.num_envs, 2, device=self.device)
        self._cmd_ang_vel = torch.zeros(self.num_envs, 1, device=self.device)
        self._cmd_resample_intervals = torch.zeros(self.num_envs, 1, device=self.device)
        self._cmd_resample_accums = torch.zeros(self.num_envs, 1, device=self.device)
        if not self.cfg.is_second_stage:
            self._cmd_commands = torch.tensor([2.5, 1.5, 3.0], device=self.device)
            self._cmd_not_zero_out_prob = torch.tensor([0.8, 0.5, 0.5], device=self.device)
        else:
            self._cmd_commands = torch.tensor([2.5, 1.5, 2.0], device=self.device)
            self._cmd_not_zero_out_prob = torch.tensor([0.8, 0.5, 0.5], device=self.device)
        self._reset_commands()

        self._show_debug_viz = False
        if self.cfg.is_play_env:
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

            goal_vel_viz_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_goal")
            # cur_vel_viz_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_current")
            goal_vel_viz_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
            # cur_vel_viz_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

            self._goal_vel_viz = VisualizationMarkers(cfg=goal_vel_viz_cfg)
            # self._cur_vel_viz = VisualizationMarkers(cfg=cur_vel_viz_cfg)
            self._goal_vel_viz.set_visibility(True)
            # self._cur_vel_viz.set_visibility(True)

        self._update_debug_draw()

        self._track_env_id = 0
        asyncio.ensure_future(self._setup_ui())

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.num_terrain_rows = 10
        self.num_terrain_cols = 20

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.cfg.terrain.terrain_generator.num_rows = self.num_terrain_rows
        self.cfg.terrain.terrain_generator.num_cols = self.num_terrain_cols
        self._terrain: TerrainImporter = self.cfg.terrain.class_type(self.cfg.terrain)

        rand_cols = torch.randint(0, self.num_terrain_cols, size=(self.num_envs,), device=self.device)
        self._env_terrain_cols = rand_cols
        self._terrain.env_origins[:] = self._terrain.terrain_origins[0, rand_cols]
        self._terrain.terrain_levels[:] = 0

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        sky_light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        )
        sky_light_cfg.func("/World/skyLight", sky_light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._step_counter += 1

        self._update_debug_draw()

        if self.cfg.is_play_env:
            lookat_pos = self._robot.data.root_pos_w[self._track_env_id].cpu()
            eye_pos = lookat_pos.clone()
            eye_pos[0] += 2.0
            eye_pos[1] += 2.0
            eye_pos[2] += 1.0
            self.viewport_camera_controller.update_view_location(eye=eye_pos, lookat=lookat_pos)

        self._previous_actions.append(self._actions.clone())
        self._previous_actions.pop(0)
        self._actions = actions.clone()

        self._cmd_resample_accums += self.step_dt
        self._reset_commands(self._cmd_resample_accums >= self._cmd_resample_intervals)

    def _apply_action(self):
        action = self._actions * 0.8 + self._previous_actions[-1] * 0.2
        action_scaled = action * self.cfg.action_scale
        action_scaled[:, self._hip_ids_jt] *= 0.5
        joint_pos_target = action_scaled + self._robot.data.default_joint_pos
        self._robot.set_joint_position_target(joint_pos_target)

    def _get_observations(self) -> dict:
        contact_indicators = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_cs, :], dim=-1) > 0.1
        contact_indicators = contact_indicators.float()

        obs_buf = torch.cat(
            [
                self._robot.data.root_com_ang_vel_b * 0.25,  # 3
                self._robot.data.projected_gravity_b,  # 3
                self._cmd_lin_vel * 2.0,  # 2
                self._cmd_ang_vel * 0.25,  # 1
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,  # 12
                self._robot.data.joint_vel * 0.05,  # 12
                self._actions,  # 12
            ],
            dim=-1,
        )

        critic_obs_buf = torch.cat(
            [
                obs_buf,
                self._robot.data.root_com_lin_vel_b * 2.0,
                contact_indicators,
                self._robot.data.applied_torque,
                self._robot.data.body_lin_vel_w[:, self._feet_ids_bd, :].reshape(self.num_envs, -1),
                self._robot.data.body_pos_w[:, self._feet_ids_bd, :].reshape(self.num_envs, -1),
            ],
            dim=-1,
        )

        noise_buf = torch.cat(
            [
                torch.ones(3) * 0.2,  # angular velocity
                torch.ones(3) * 0.05,  # projected gravity
                torch.zeros(3),  # commands
                torch.ones(12) * 0.01,  # joint positions
                torch.ones(12) * 1.5 * 0.5,  # joint velocities
                torch.zeros(self._num_actions),  # actions
            ],
            dim=0,
        )
        obs_buf += (torch.rand_like(obs_buf) * 2.0 - 1.0) * noise_buf.to(self.device)

        return {"policy": obs_buf, "critic": critic_obs_buf}

    def _get_rewards(self) -> torch.Tensor:
        r_lin_vel_z_l2 = torch.square(self._robot.data.root_com_lin_vel_b[:, 2])
        r_lin_vel_z_l2 *= -2.0

        r_ang_vel_xy_l2 = torch.sum(torch.square(self._robot.data.root_com_ang_vel_b[:, :2]), dim=1)
        r_ang_vel_xy_l2 *= -0.05

        r_flat_orientation_l2 = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        r_flat_orientation_l2 *= -5.0

        r_dof_torques_l2 = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        r_dof_torques_l2 *= -0.0002

        r_power = torch.sum(torch.abs(self._robot.data.applied_torque) * torch.abs(self._robot.data.joint_vel), dim=1)
        r_power *= -0.001

        r_action_rate_l2 = torch.sum(torch.square(self._actions - self._previous_actions[-1]), dim=1)
        r_action_rate_l2 *= -0.01

        out_of_limits = -(self._robot.data.joint_pos - self._robot.data.soft_joint_pos_limits[:, :, 0]).clip(max=0.0)
        out_of_limits += (self._robot.data.joint_pos - self._robot.data.soft_joint_pos_limits[:, :, 1]).clip(min=0.0)
        r_dof_pos_limits = torch.sum(out_of_limits, dim=1)
        r_dof_pos_limits *= -10.0

        out_of_limits_vel = (self._robot.data.joint_vel - self._robot.data.soft_joint_vel_limits).clip(min=0.0)
        r_dof_vel_limits = torch.sum(out_of_limits_vel, dim=1)
        r_dof_vel_limits *= -10.0

        r_track_lin_vel = torch.sum(
            torch.square(self._cmd_lin_vel[:, :2] - self._robot.data.root_com_lin_vel_b[:, :2]), dim=1
        )
        r_track_lin_vel = torch.exp(-r_track_lin_vel * 4.0)
        r_track_lin_vel *= 2.0

        r_track_ang_vel = torch.square(self._cmd_ang_vel[:, 0] - self._robot.data.root_ang_vel_b[:, 2])
        r_track_ang_vel = torch.exp(-r_track_ang_vel * 4.0)
        r_track_ang_vel *= 1.0

        r_dof_pos = torch.sum(torch.square(self._robot.data.joint_pos - self._robot.data.default_joint_pos), dim=1)
        r_dof_pos *= -1.0

        r_alive = torch.ones(self.num_envs, device=self.device)
        r_alive *= 0.2

        cmd_norm = torch.linalg.norm(torch.concat([self._cmd_lin_vel[:, :2], self._cmd_ang_vel], dim=-1), dim=1)
        is_trotting = (
            (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_cs[0], :], dim=-1) > 0.5)
            & (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_cs[3], :], dim=-1) > 0.5)
            & (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_cs[1], :], dim=-1) < 0.5)
            & (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_cs[2], :], dim=-1) < 0.5)
        ) | (
            (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_cs[1], :], dim=-1) > 0.5)
            & (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_cs[2], :], dim=-1) > 0.5)
            & (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_cs[0], :], dim=-1) < 0.5)
            & (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_cs[3], :], dim=-1) < 0.5)
        )
        r_trot = is_trotting * 1.0
        r_trot[cmd_norm < 0.1] = 0.0

        r_termination = self._reset_buf.float()
        r_termination *= -200.0

        if self.cfg.is_second_stage:
            is_standing = (
                (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_cs[0], :], dim=-1) > 0.5)
                & (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_cs[3], :], dim=-1) > 0.5)
                & (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_cs[1], :], dim=-1) > 0.5)
                & (torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_cs[2], :], dim=-1) > 0.5)
            )
            r_stand_still = (cmd_norm < 0.2).float() * (~is_standing).float() * -10.0

        mean_terrain_level = torch.mean(self._terrain.terrain_levels.float())
        if mean_terrain_level >= 1.0:
            r_lin_vel_z_l2 *= 5.0

            cmd_norm = torch.linalg.norm(torch.concat([self._cmd_lin_vel[:, :2], self._cmd_ang_vel], dim=-1), dim=1)
            contact = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_cs, :], dim=-1) > 0.05
            feet_vel = self._robot.data.body_lin_vel_w[:, self._feet_ids_bd, :]
            vel_xy = feet_vel[..., :2]
            vel_xy_norm_sq = torch.sum(torch.square(vel_xy), dim=-1)
            r_feet_slip = torch.sum(vel_xy_norm_sq * contact, dim=-1) * (cmd_norm > 0.01)
            r_feet_slip *= -1.0

            max_foot_height = 0.15

            feet_vel = self._robot.data.body_lin_vel_w[:, self._feet_ids_bd, :]
            vel_xy = feet_vel[..., :2]
            vel_norm = torch.sqrt(torch.norm(vel_xy, dim=-1))
            foot_pos = self._robot.data.body_pos_w[:, self._feet_ids_bd, :]
            foot_z = foot_pos[..., -1]
            delta = torch.abs(foot_z - max_foot_height)
            r_feet_clearance = torch.sum(delta * vel_norm, dim=-1)
            r_feet_clearance *= -2.0

            first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids_cs]
            self._swing_peak = torch.maximum(self._swing_peak, foot_z)
            error = self._swing_peak / max_foot_height - 1.0
            r_feet_height = torch.max(torch.square(error) * first_contact, dim=-1).values * (cmd_norm > 0.01)
            r_feet_height *= -2.0
            self._swing_peak *= ~contact

            last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids_cs]
            r_feet_air_time = torch.sum((last_air_time - 0.15) * first_contact, dim=1) * (cmd_norm > 0.01)
            r_feet_air_time *= 10.0

        rewards = {k: v * self.step_dt for k, v in locals().items() if k.startswith("r_")}

        for k, v in rewards.items():
            if k not in self._episode_sums:
                self._episode_sums[k] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            if v.shape != self._episode_sums[k].shape:
                raise ValueError(f"reward {k} has wrong shape: {v.shape}, expected: {self._episode_sums[k].shape}")
            self._episode_sums[k] += v

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        reward.clip_(min=0.0)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id_cs], dim=-1), dim=1)[0] > 1.0,
            dim=1,
        )
        died |= -self._robot.data.projected_gravity_b[:, 2] < 0.25
        self._reset_buf[:] = died

        if torch.any(self._robot.data.root_pos_w[:, 2] < -5.0):
            raise RuntimeError("robot is falling down!")

        return died, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        env_mask[env_ids] = True

        if not self._first_reset:
            distance = torch.norm(
                self._robot.data.root_pos_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2], dim=1
            )
            move_up = distance > self.cfg.terrain.terrain_generator.size[0] * 0.5
            move_down = distance < 1.0
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

        if self.cfg.is_play_env:
            self._terrain.terrain_levels[0] = self.num_terrain_rows // 2
            self._env_terrain_cols[0] = self.num_terrain_cols // 2
            self._terrain.env_origins[env_ids, :] = self._terrain.terrain_origins[
                self._terrain.terrain_levels[env_ids], self._env_terrain_cols[env_ids], :
            ]

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        for actions in self._previous_actions:
            actions[env_ids] = 0.0

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        zeros = torch.zeros(default_root_state.shape[0], device=self.device)
        rand_yaw = torch.rand_like(zeros) * math.pi * 2.0 - math.pi
        rand_quats = math_utils.quat_from_euler_xyz(zeros, zeros, rand_yaw)
        default_root_state[:, 3:7] = math_utils.quat_mul(default_root_state[:, 3:7], rand_quats)
        self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        default_joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        default_joint_pos *= math_utils.sample_uniform(0.6, 1.4, default_joint_pos.shape, default_joint_pos.device)
        joint_pos_limits = self._robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = default_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        self._robot.write_joint_state_to_sim(joint_pos, self._robot.data.default_joint_vel[env_ids], env_ids=env_ids)

        self._reset_commands(env_mask)

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Episode/average_speed"] = torch.mean(self._robot.data.root_com_lin_vel_b.norm(dim=1))
        extras["Episode/average_terrain_level"] = torch.mean(self._terrain.terrain_levels.to(torch.float))
        extras["Episode/max_terrain_level"] = torch.max(self._terrain.terrain_levels.to(torch.float))
        self.extras["log"] = extras

    def _reset_commands(self, masks: torch.Tensor | None = None):
        if masks is None:
            masks = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        masks = masks.squeeze()

        num_resets = torch.count_nonzero(masks).item()
        if num_resets == 0:
            return

        new_commands = math_utils.sample_uniform(
            -self._cmd_commands, self._cmd_commands, (num_resets, 3), device=self.device
        )
        new_commands[:, 0] = new_commands[:, 0].abs()
        zero_out = (
            math_utils.sample_uniform(0.0, 1.0, (num_resets, 3), device=self.device)
            > self._cmd_not_zero_out_prob[None, :]
        )
        new_commands[zero_out] = 0.0
        self._cmd_lin_vel[masks, :2] = new_commands[:, :2]
        self._cmd_ang_vel[masks, 0] = new_commands[:, 2]

        self._cmd_resample_intervals[masks] = math_utils.sample_uniform(
            self.cfg.cmd_resample_interval[0], self.cfg.cmd_resample_interval[1], (num_resets, 1), self.device
        )
        self._cmd_resample_accums[masks] = torch.zeros(num_resets, 1, device=self.device)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self._goal_vel_viz.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self._robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat

    def _update_debug_draw(self):
        if not self._show_debug_viz:
            return

        self._debug_draw.clear_lines()
        self._debug_draw.clear_points()

        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._cmd_lin_vel[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self._robot.data.root_lin_vel_b[:, :2])
        self._goal_vel_viz.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        # self._cur_vel_viz.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

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

    def _randomize_mass(self, min_delta, max_delta):
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
            min_delta, max_delta, (masses.shape[0], body_ids.shape[0]), device=masses.device
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
        print(f"min mass: {torch.min(new_masses[env_ids, body_ids])}")
        print(f"max mass: {torch.max(new_masses[env_ids, body_ids])}")
        print("*" * 50)

    async def _setup_ui(self):
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

        def inc_cmd_x():
            self._cmd_lin_vel[self._track_env_id, 0] += 0.5
            print(f"Env {self._track_env_id} cmd x velocity: {self._cmd_lin_vel[self._track_env_id, 0].item():.2f}")

        def dec_cmd_x():
            self._cmd_lin_vel[self._track_env_id, 0] -= 0.5
            print(f"Env {self._track_env_id} cmd x velocity: {self._cmd_lin_vel[self._track_env_id, 0].item():.2f}")

        def inc_cmd_y():
            self._cmd_lin_vel[self._track_env_id, 1] += 0.5
            print(f"Env {self._track_env_id} cmd y velocity: {self._cmd_lin_vel[self._track_env_id, 1].item():.2f}")

        def dec_cmd_y():
            self._cmd_lin_vel[self._track_env_id, 1] -= 0.5
            print(f"Env {self._track_env_id} cmd y velocity: {self._cmd_lin_vel[self._track_env_id, 1].item():.2f}")

        def inc_cmd_w():
            self._cmd_ang_vel[self._track_env_id, 0] += 0.5
            print(f"Env {self._track_env_id} cmd ang vel: {self._cmd_ang_vel[self._track_env_id, 0].item():.2f}")

        def dec_cmd_w():
            self._cmd_ang_vel[self._track_env_id, 0] -= 0.5
            print(f"Env {self._track_env_id} cmd ang vel: {self._cmd_ang_vel[self._track_env_id, 0].item():.2f}")

        def reset_cmd():
            self._cmd_lin_vel[self._track_env_id, 0] = 0.0
            self._cmd_lin_vel[self._track_env_id, 1] = 0.0
            self._cmd_ang_vel[self._track_env_id, 0] = 0.0
            print(f"Env {self._track_env_id} commands reset to zero.")

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
                ui.Label("Modify Velocity Commands", style={"color": 0xFFFFFFFF, "font_size": 16})
                with ui.HStack():
                    ui.Button("Inc X", clicked_fn=inc_cmd_x)
                    ui.Button("Dec X", clicked_fn=dec_cmd_x)
                with ui.HStack():
                    ui.Button("Inc Y", clicked_fn=inc_cmd_y)
                    ui.Button("Dec Y", clicked_fn=dec_cmd_y)
                with ui.HStack():
                    ui.Button("Inc W", clicked_fn=inc_cmd_w)
                    ui.Button("Dec W", clicked_fn=dec_cmd_w)
                ui.Button("Reset Commands", clicked_fn=reset_cmd)
