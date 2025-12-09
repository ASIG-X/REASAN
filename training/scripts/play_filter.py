# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse
import traceback

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=3)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--collect", type=str, default="none", choices=["none", "train", "val", "train_extra", "val_extra"])
parser.add_argument("--use_pred_rays", action="store_true", default=False)
parser.add_argument("--keyboard", action="store_true", default=False)
parser.add_argument("--no_obstacle", action="store_true", default=False)
parser.add_argument("--confirm", action="store_true", default=False)
parser.add_argument("--with_dyn_obst", action="store_true", default=False)
parser.add_argument("--obst_speed_range", type=float, nargs=2, default=(0.5, 1.5))
parser.add_argument("--num_ray_centers", type=str, default="11x", choices=["1x", "3x", "5x", "11x"])

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# set task name
task_name = "Unitree-Go2-Filter"
print("=" * 50)
print("Playing filter policy.")
print("=" * 50)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Start playing after the app is launched."""

import os
import time

import go2_lidar.tasks  # noqa: F401
import gymnasium as gym
import torch
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.modules.actor_critic_recurrent_ray import export_actor_onnx, export_actor_torchscript
from rsl_rl.runners import OnPolicyRunner


def main():
    # parse configuration
    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"resume path: {resume_path}")
    print(f"log dir: {log_dir}")

    # update env cfg
    env_cfg.is_play_env = True
    env_cfg.events.push_robot = None
    env_cfg.data_collection_type = args_cli.collect
    env_cfg.use_predicted_rays = args_cli.use_pred_rays
    env_cfg.use_keyboard = args_cli.keyboard
    env_cfg.no_obstacle = args_cli.no_obstacle
    env_cfg.wait_for_key = not args_cli.confirm
    env_cfg.use_dynamic_obstacle = args_cli.with_dyn_obst
    env_cfg.obst_speed_range = args_cli.obst_speed_range
    env_cfg.set_raycaster_measure_pattern(args_cli.num_ray_centers)

    # create the environment
    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load previously trained model
    ppo_runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
        use_ray=True,
    )
    ppo_runner.load(resume_path, load_optimizer=False)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    print("\n\nExporting the policy: [ONNX]")
    try:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        if not os.path.exists(export_model_dir):
            os.makedirs(export_model_dir)
        print(f"trying to export policy to onnx: {export_model_dir}")
        export_actor_onnx(ppo_runner.alg.policy, onnx_path=os.path.join(export_model_dir, "policy.onnx"))
    except Exception:
        print("*" * 50)
        print("failed to export policy to onnx.")
        print("*" * 50)
        traceback.print_exc()
        print("*" * 50)

    # export policy to torchscript
    print("\n\nExporting the policy: [TorchScript]")
    try:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        if not os.path.exists(export_model_dir):
            os.makedirs(export_model_dir)
        print(f"trying to export policy to torchscript: {export_model_dir}")
        export_actor_torchscript(ppo_runner.alg.policy, torchscript_path=os.path.join(export_model_dir, "policy.pt"))
    except Exception:
        print("*" * 50)
        print("failed to export policy to torchscript.")
        print("*" * 50)
        traceback.print_exc()
        print("*" * 50)

    obs, _ = env.get_observations()
    timestep = 0
    dt = env.unwrapped.step_dt

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
