import gymnasium as gym

# flake8: noqa F401
from . import (
    go2_filter_env,
    go2_filter_ppo_cfg,
    go2_loco_env,
    go2_loco_ppo_cfg,
    go2_nav_env,
    go2_nav_ppo_cfg,
)

gym.register(
    id="Unitree-Go2-Locomotion",
    entry_point=f"{__name__}.go2_loco_env:Go2LocoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_loco_env:Go2LocoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.go2_loco_ppo_cfg:Go2LocoPPOCfg",
    },
)

gym.register(
    id="Unitree-Go2-Filter",
    entry_point=f"{__name__}.go2_filter_env:Go2FilterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_filter_env:Go2FilterEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.go2_filter_ppo_cfg:Go2FilterPPOCfg",
    },
)

gym.register(
    id="Unitree-Go2-Navigation",
    entry_point=f"{__name__}.go2_nav_env:Go2NavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_nav_env:Go2NavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.go2_nav_ppo_cfg:Go2NavPPOCfg",
    },
)
