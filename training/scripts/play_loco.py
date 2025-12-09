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

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# set task name
task_name = "Unitree-Go2-Locomotion"
print("=" * 50)
print("Playing locomotion policy.")
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
from isaaclab_rl.rsl_rl.exporter import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunnerLoco


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
    ppo_runner = OnPolicyRunnerLoco(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )
    ppo_runner.load(resume_path, load_optimizer=False)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    policy_nn = ppo_runner.alg.policy

    print("\n\nExporting the policy: [ONNX]")
    try:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        if not os.path.exists(export_model_dir):
            os.makedirs(export_model_dir)
        print(f"trying to export policy to onnx: {export_model_dir}")
        export_policy_as_onnx(
            policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )
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
        export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    except Exception:
        print("*" * 50)
        print("failed to export policy to torchscript.")
        print("*" * 50)
        traceback.print_exc()
        print("*" * 50)

    obs, _ = env.get_observations()
    timestep = 0

    # simulate environment
    while simulation_app.is_running():
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

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
