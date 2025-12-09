# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets import UNITREE_GO2_CFG

from go2_lidar.motor.random_dc_motor import RandomDCMotorCfg
from go2_lidar.sensor.lidar_pattern import (
    Ray3D_3x180_Parallel_1x_PatternCfg,
    Ray3D_3x180_Parallel_3x_PatternCfg,
    Ray3D_3x180_Parallel_5x_PatternCfg,
    Ray3D_3x180_Parallel_11x_PatternCfg,
    Ray3D_4x90_PatternCfg,
)
from go2_lidar.tasks.go2_loco_env_cfg import randomize_rigid_body_com
from go2_lidar.terrain.train_terrain_cfg import GO2_FILTER_TERRAIN_CFG


@configclass
class EventCfg:
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-1.0, 1.0),
            "torque_range": (-1.0, 1.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    base_com = EventTerm(
        func=randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.02, 0.02)},
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class Go2FilterEnvCfg(DirectRLEnvCfg):
    episode_length_s = 9.0
    decimation = 4

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(gpu_max_rigid_patch_count=4096 * 4096),
    )

    viewer: ViewerCfg = ViewerCfg(resolution=(1920, 1080))

    num_loco_actions = 12
    num_high_actions = 3
    action_space = num_high_actions
    action_scale_loco = 0.25
    is_play_env = False

    data_collection_type = "none"
    use_predicted_rays = False
    use_keyboard = False
    no_obstacle = False
    use_dynamic_obstacle = False
    wait_for_key = True
    obst_speed_range = (0.5, 1.5)

    state_space = 0
    loco_policy = None

    scene = InteractiveSceneCfg(num_envs=4096, env_spacing=10, replicate_physics=False)
    events: EventCfg = EventCfg()

    static_friction_range = (0.2, 1.25)
    dynamic_friction_range = (0.2, 1.25)
    restitution_range = (0.0, 1.0)

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=GO2_FILTER_TERRAIN_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
            texture_scale=(4.0, 4.0),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    @configclass
    class RayCasterCfgExtended(RayCasterCfg):
        attach_yaw_only_rotate: bool = False

    # mid360 lidar
    raycaster = RayCasterCfgExtended(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.249, 0.0, 0.135)),
        drift_range=(-0.03, 0.03),
        attach_yaw_only=False,
        pattern_cfg=Ray3D_4x90_PatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # ground truth ray caster
    raycaster_measure = RayCasterCfgExtended(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=None,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        attach_yaw_only_rotate=True,
    )
    num_ray_centers = None

    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
        track_pose=True,
    )

    def __post_init__(self):
        self._build_observation_space()

        self.sim.render_interval = self.decimation

        self.raycaster.update_period = self.sim.dt * self.decimation
        self.raycaster_measure.update_period = self.sim.dt * self.decimation

        self.robot.actuators["base_legs"] = RandomDCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
            motor_strengths_range=(0.9, 1.1),
        )

    def _build_observation_space(self):
        self.obs_ranges = {}
        self.obs_dims = {}
        start_idx = 0

        obs_proprio_dim = 12
        self.obs_dims["proprio"] = obs_proprio_dim
        self.obs_ranges["proprio"] = (0, obs_proprio_dim)
        start_idx += obs_proprio_dim

        obs_priv_dim = 6
        self.obs_dims["priv"] = obs_priv_dim
        self.obs_ranges["priv"] = (start_idx, start_idx + obs_priv_dim)
        start_idx += obs_priv_dim

        obs_actor_ray = 180
        self.obs_dims["actor_ray"] = obs_actor_ray
        self.obs_ranges["actor_ray"] = (start_idx, start_idx + obs_actor_ray)
        start_idx += obs_actor_ray

        obs_critic_ray_dim = 540
        self.obs_dims["critic_ray"] = obs_critic_ray_dim
        self.obs_ranges["critic_ray"] = (start_idx, start_idx + obs_critic_ray_dim)
        start_idx += obs_critic_ray_dim

        self.observation_space = start_idx

    def set_raycaster_measure_pattern(self, pattern_name: str):
        if pattern_name == "1x":
            self.raycaster_measure.pattern_cfg = Ray3D_3x180_Parallel_1x_PatternCfg()
            self.num_ray_centers = 1
        elif pattern_name == "3x":
            self.raycaster_measure.pattern_cfg = Ray3D_3x180_Parallel_3x_PatternCfg()
            self.num_ray_centers = 3
        elif pattern_name == "5x":
            self.raycaster_measure.pattern_cfg = Ray3D_3x180_Parallel_5x_PatternCfg()
            self.num_ray_centers = 5
        elif pattern_name == "11x":
            self.raycaster_measure.pattern_cfg = Ray3D_3x180_Parallel_11x_PatternCfg()
            self.num_ray_centers = 11
        else:
            raise ValueError(f"Invalid pattern name: {pattern_name}")
