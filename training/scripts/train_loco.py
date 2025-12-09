# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=10000)
parser.add_argument("--gui", action="store_true")
parser.add_argument("--wandb_proj", type=str, default=None)
parser.add_argument("--second_stage", action="store_true", default=False)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# --gui takes precedence over --headless
args_cli.headless = not args_cli.gui

# set task name
task_name = "Unitree-Go2-Locomotion"
print("=" * 50)
print("Training locomotion policy.")
print("=" * 50)

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""After the app is launched, start training."""

import os
from datetime import datetime

import go2_lidar.tasks  # noqa: F401
import gymnasium as gym
import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunnerLoco

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    # load configs from the registry
    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    # set number of environments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set maximum iterations
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    if args_cli.second_stage:
        env_cfg.set_second_stage()

    # set device: default cuda:0
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # set run name
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir = agent_cfg.run_name

    print(f"Exact experiment name requested from command line: {log_dir}")
    log_dir = os.path.join(log_root_path, log_dir)

    # update env config
    env_cfg.is_play_env = False

    # create environment
    env = gym.make(task_name, cfg=env_cfg, render_mode=None)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print("=" * 50)
        print(f"Loading checkpoint from: {resume_path}")
        print("=" * 50)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunnerLoco(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
        wandb_project=args_cli.wandb_proj,
    )

    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations)

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
