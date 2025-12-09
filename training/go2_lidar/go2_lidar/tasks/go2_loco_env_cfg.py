# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets import UNITREE_GO2_CFG

from go2_lidar.motor.random_dc_motor import RandomDCMotorCfg
from go2_lidar.terrain.train_terrain_cfg import GO2_LOCO_TERRAIN_CFG


def randomize_rigid_body_com(
    env,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.
    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), len(body_ids), 3), device="cpu")
    rand_samples = rand_samples.squeeze()

    # get the current com of the bodies
    coms = asset.root_physx_view.get_coms().clone()
    coms[env_ids, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids.unsqueeze(1))

    print("*" * 50)
    print("Randomized CoM:")
    print(coms[env_ids, body_ids, :3])
    print("*" * 50)


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
class Go2LocoEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
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

    observation_space = 45
    state_space = 0

    action_space = 12
    action_scale = 0.25
    is_play_env = False
    is_second_stage = False

    cmd_resample_interval = (4.0, 4.0)

    scene = InteractiveSceneCfg(num_envs=4096, env_spacing=10, replicate_physics=False)
    events: EventCfg = EventCfg()

    static_friction_range = (0.2, 1.25)
    dynamic_friction_range = (0.2, 1.25)
    restitution_range = (0.0, 1.0)

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=GO2_LOCO_TERRAIN_CFG,
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

    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
        track_pose=True,
    )

    def __post_init__(self):
        self.sim.render_interval = self.decimation

        self.contact_sensor.update_period = self.sim.dt

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

    def set_second_stage(self):
        self.is_second_stage = True
        self.events.push_robot.interval_range_s = (2.0, 2.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}
        self.cmd_resample_interval = (2.0, 3.0)
