# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RslRlPpoActorCriticHierarchicalCfg(RslRlPpoActorCriticCfg):
    loco_class_name: str = MISSING


@configclass
class Go2FilterPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 200
    experiment_name = "go2_lidar"
    empirical_normalization = False
    policy = RslRlPpoActorCriticHierarchicalCfg(
        class_name="ActorCriticRecurrentRay",
        loco_class_name="ActorCriticRecurrent",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=10,
        num_mini_batches=8,
        learning_rate=0.001,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )
