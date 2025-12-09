# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.sub_terrain_cfg import FlatPatchSamplingCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from go2_lidar.terrain.hf_terrains_cfg import HfRandomUniformTerrainCfg
from go2_lidar.terrain.nav_terrain_generator import NavTerrainGenerator
from go2_lidar.terrain.test_terrain_generator import TestTerrainGenerator

GO2_LOCO_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    curriculum=True,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.01, 0.03), noise_step=0.01, border_width=0.1
        ),
    },
)

GO2_FILTER_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=0,
    size=(9.0, 9.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "random_rough": HfRandomUniformTerrainCfg(
            terrain_variant="filter",
            proportion=1.0,
            noise_range=(0.01, 0.03),
            noise_step=0.01,
            border_width=0.5,
            flat_patch_sampling={
                "patches": FlatPatchSamplingCfg(
                    num_patches=20, patch_radius=[i * 0.1 for i in range(10)], max_height_diff=0.1
                )
            },
        ),
    },
)

GO2_NAV_TERRAIN_CFG = TerrainGeneratorCfg(
    class_type=NavTerrainGenerator,
    seed=0,
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,  # number of curriculum levels
    num_cols=30,  # number of terrains per curriculum level
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "random_rough": HfRandomUniformTerrainCfg(
            terrain_variant="nav",
            proportion=1.0,
            noise_range=(0.01, 0.03),
            noise_step=0.01,
            border_width=0.1,
        ),
    },
)

GO2_TEST_TERRAIN_CFG = TerrainGeneratorCfg(
    class_type=TestTerrainGenerator,
    seed=0,
    size=(20.0, 10.0),
    border_width=0.0,
    num_rows=10,  # number of variants per testing case
    num_cols=8,  # number of testing cases
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "random_rough": HfRandomUniformTerrainCfg(
            terrain_variant="test",
            proportion=1.0,
            noise_range=(0.01, 0.03),
            noise_step=0.01,
            border_width=0.1,
        ),
    },
)

GO2_ABS_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=0,
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    curriculum=True,
    sub_terrains={
        "random_rough": HfRandomUniformTerrainCfg(
            terrain_variant="abs",
            proportion=1.0,
            noise_range=(0.01, 0.03),
            noise_step=0.01,
            border_width=0.1,
            flat_patch_sampling={
                "patches": FlatPatchSamplingCfg(
                    num_patches=20, patch_radius=[i * 0.1 for i in range(10)], max_height_diff=0.1
                )
            },
        ),
    },
)
