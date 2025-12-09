# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast sensor."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import MISSING
from typing import Literal

import torch
from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
from isaaclab.utils import configclass


def ray_3d_pattern(cfg: Ray3DPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Lidar sensor pattern for ray casting.

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays.
    """
    # define angles
    vertical_angles = torch.tensor([-3.5, 0.0, 7.5, 15.0, 22.5, 30.0, 37.5, 45.0])
    num_horizontal_angles = torch.tensor([90, 90, 90, 90, 90, 90, 90, 90], dtype=torch.int32)

    # Convert degrees to radians
    vertical_angles_rad = torch.deg2rad(vertical_angles)
    horizontal_angles_rad = [
        torch.deg2rad(torch.linspace(-180.0, 180.0 - 360.0 / num_horizontal_angles, num_horizontal_angles))
        for num_horizontal_angles in num_horizontal_angles
    ]

    # create grid
    v_angles = []
    for i in range(len(vertical_angles)):
        v_angles.append(vertical_angles_rad[i] * torch.ones_like(horizontal_angles_rad[i]))
    v_angles = torch.cat(v_angles, dim=0)
    h_angles = torch.cat(horizontal_angles_rad, dim=0)

    # Spherical to Cartesian conversion (assuming Z is up)
    x = torch.cos(v_angles) * torch.cos(h_angles)
    y = torch.cos(v_angles) * torch.sin(h_angles)
    z = torch.sin(v_angles)

    # Ray directions
    ray_directions = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(device)

    # Ray starts: Assuming all rays originate from (0,0,0)
    ray_starts = torch.zeros_like(ray_directions).to(device)

    return ray_starts, ray_directions


def ray_2d_pattern(cfg: Ray3DPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Lidar sensor pattern for ray casting.

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays.
    """
    # define angles
    vertical_angles = torch.tensor([0.0])
    num_horizontal_angles = torch.tensor([90], dtype=torch.int32)

    # Convert degrees to radians
    vertical_angles_rad = torch.deg2rad(vertical_angles)
    horizontal_angles_rad = [
        torch.deg2rad(torch.linspace(-180.0, 180.0 - 360.0 / num_horizontal_angles, num_horizontal_angles))
        for num_horizontal_angles in num_horizontal_angles
    ]

    # create grid
    v_angles = []
    for i in range(len(vertical_angles)):
        v_angles.append(vertical_angles_rad[i] * torch.ones_like(horizontal_angles_rad[i]))
    v_angles = torch.cat(v_angles, dim=0)
    h_angles = torch.cat(horizontal_angles_rad, dim=0)

    # Spherical to Cartesian conversion (assuming Z is up)
    x = torch.cos(v_angles) * torch.cos(h_angles)
    y = torch.cos(v_angles) * torch.sin(h_angles)
    z = torch.sin(v_angles)

    # Ray directions
    ray_directions = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(device)

    # Ray starts: Assuming all rays originate from (0,0,0)
    ray_starts = torch.zeros_like(ray_directions).to(device)

    return ray_starts, ray_directions


def ray_3d_45_90_45_30_pattern(cfg: Ray3DPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Lidar sensor pattern for ray casting.

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays.
    """
    # define angles
    vertical_angles = torch.tensor([-5.0, 0.0, 10.0, 20.0])
    num_horizontal_angles = torch.tensor([45, 90, 45, 30], dtype=torch.int32)

    # Convert degrees to radians
    vertical_angles_rad = torch.deg2rad(vertical_angles)
    horizontal_angles_rad = [
        torch.deg2rad(torch.linspace(-180.0, 180.0 - 360.0 / num_horizontal_angles, num_horizontal_angles))
        for num_horizontal_angles in num_horizontal_angles
    ]

    # create grid
    v_angles = []
    for i in range(len(vertical_angles)):
        v_angles.append(vertical_angles_rad[i] * torch.ones_like(horizontal_angles_rad[i]))
    v_angles = torch.cat(v_angles, dim=0)
    h_angles = torch.cat(horizontal_angles_rad, dim=0)

    # Spherical to Cartesian conversion (assuming Z is up)
    x = torch.cos(v_angles) * torch.cos(h_angles)
    y = torch.cos(v_angles) * torch.sin(h_angles)
    z = torch.sin(v_angles)

    # Ray directions
    ray_directions = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(device)

    # Ray starts: Assuming all rays originate from (0,0,0)
    ray_starts = torch.zeros_like(ray_directions).to(device)

    return ray_starts, ray_directions


def ray_3d_4x90_pattern(cfg: Ray3DPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Lidar sensor pattern for ray casting.

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays.
    """
    # define angles
    vertical_angles = torch.tensor([-5.0, 0.0, 10.0, 20.0])
    num_horizontal_angles = torch.tensor([90, 90, 90, 90], dtype=torch.int32)

    # Convert degrees to radians
    vertical_angles_rad = torch.deg2rad(vertical_angles)
    horizontal_angles_rad = [
        torch.deg2rad(torch.linspace(-180.0, 180.0 - 360.0 / num_horizontal_angles, num_horizontal_angles))
        for num_horizontal_angles in num_horizontal_angles
    ]

    # create grid
    v_angles = []
    for i in range(len(vertical_angles)):
        v_angles.append(vertical_angles_rad[i] * torch.ones_like(horizontal_angles_rad[i]))
    v_angles = torch.cat(v_angles, dim=0)
    h_angles = torch.cat(horizontal_angles_rad, dim=0)

    # Spherical to Cartesian conversion (assuming Z is up)
    x = torch.cos(v_angles) * torch.cos(h_angles)
    y = torch.cos(v_angles) * torch.sin(h_angles)
    z = torch.sin(v_angles)

    # Ray directions
    ray_directions = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(device)

    # Ray starts: Assuming all rays originate from (0,0,0)
    ray_starts = torch.zeros_like(ray_directions).to(device)

    return ray_starts, ray_directions


def ray_3d_3x180_parallel_pattern(cfg: Ray3DPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Lidar sensor pattern for ray casting.

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays.
    """
    # define heights
    vertical_offsets = torch.tensor([0.0, 0.1, 0.2])
    # define angles
    num_horizontal_angles = torch.tensor([180, 180, 180], dtype=torch.int32)

    # Convert degrees to radians
    horizontal_angles_rad = [
        torch.deg2rad(torch.linspace(-180.0, 180.0 - 360.0 / num_horizontal_angles, num_horizontal_angles))
        for num_horizontal_angles in num_horizontal_angles
    ]

    # create grid
    v_angles = [torch.tensor([0] * x) for x in num_horizontal_angles]
    v_angles = torch.cat(v_angles, dim=0)
    h_angles = torch.cat(horizontal_angles_rad, dim=0)

    # Spherical to Cartesian conversion (assuming Z is up)
    x = torch.cos(v_angles) * torch.cos(h_angles)
    y = torch.cos(v_angles) * torch.sin(h_angles)
    z = torch.sin(v_angles)

    # Ray directions
    ray_directions = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(device)

    # Ray starts: Assuming all rays originate from (0,0,0)
    ray_starts = torch.zeros_like(ray_directions).to(device)
    ray_starts[:, 2] = (
        torch.tensor([[vertical_offsets[i]] * num_horizontal_angles[i] for i in range(len(vertical_offsets))])
        .flatten()
        .to(device)
    )

    return ray_starts, ray_directions


def ray_3d_3x180_parallel_1x_pattern(cfg: Ray3DPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Lidar sensor pattern for ray casting.

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays.
    """
    # define heights
    vertical_offsets = torch.tensor([0.0, 0.1, 0.2])
    # define angles
    num_horizontal_angles = torch.tensor([180, 180, 180], dtype=torch.int32)

    # Convert degrees to radians
    horizontal_angles_rad = [
        torch.deg2rad(torch.linspace(-180.0, 180.0 - 360.0 / num_horizontal_angles, num_horizontal_angles))
        for num_horizontal_angles in num_horizontal_angles
    ]

    # create grid
    v_angles = [torch.tensor([0] * x) for x in num_horizontal_angles]
    v_angles = torch.cat(v_angles, dim=0)
    h_angles = torch.cat(horizontal_angles_rad, dim=0)

    # Spherical to Cartesian conversion (assuming Z is up)
    x = torch.cos(v_angles) * torch.cos(h_angles)
    y = torch.cos(v_angles) * torch.sin(h_angles)
    z = torch.sin(v_angles)

    # Ray directions
    ray_directions = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(device)

    # Ray starts: Assuming all rays originate from (0,0,0)
    ray_starts = torch.zeros_like(ray_directions).to(device)
    ray_starts[:, 2] = (
        torch.tensor([[vertical_offsets[i]] * num_horizontal_angles[i] for i in range(len(vertical_offsets))])
        .flatten()
        .to(device)
    )

    return ray_starts, ray_directions


def ray_3d_3x180_parallel_3x_pattern(cfg: Ray3DPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Lidar sensor pattern for ray casting.

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays.
    """
    # define heights
    vertical_offsets = torch.tensor([0.0, 0.1, 0.2])
    # define angles
    num_horizontal_angles = torch.tensor([180, 180, 180], dtype=torch.int32)

    # Convert degrees to radians
    horizontal_angles_rad = [
        torch.deg2rad(torch.linspace(-180.0, 180.0 - 360.0 / num_horizontal_angles, num_horizontal_angles))
        for num_horizontal_angles in num_horizontal_angles
    ]

    # create grid
    v_angles = [torch.tensor([0] * x) for x in num_horizontal_angles]
    v_angles = torch.cat(v_angles, dim=0)
    h_angles = torch.cat(horizontal_angles_rad, dim=0)

    # Spherical to Cartesian conversion (assuming Z is up)
    x = torch.cos(v_angles) * torch.cos(h_angles)
    y = torch.cos(v_angles) * torch.sin(h_angles)
    z = torch.sin(v_angles)

    # Ray directions
    ray_directions = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(device)

    # Ray starts: Assuming all rays originate from (0,0,0)
    ray_starts = torch.zeros_like(ray_directions).to(device)
    ray_starts[:, 2] = (
        torch.tensor([[vertical_offsets[i]] * num_horizontal_angles[i] for i in range(len(vertical_offsets))])
        .flatten()
        .to(device)
    )

    # three ray origins
    ray_directions = torch.cat([ray_directions] * 3, dim=0)
    ray_starts = torch.cat(
        [
            ray_starts + torch.tensor([0.0, 0.0, 0.0], device=device),
            ray_starts + torch.tensor([0.4, 0.0, 0.0], device=device),
            ray_starts + torch.tensor([-0.4, 0.0, 0.0], device=device),
        ],
        dim=0,
    )

    return ray_starts, ray_directions


def ray_3d_3x180_parallel_5x_pattern(cfg: Ray3DPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Lidar sensor pattern for ray casting.

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays.
    """
    # define heights
    vertical_offsets = torch.tensor([0.0, 0.1, 0.2])
    # define angles
    num_horizontal_angles = torch.tensor([180, 180, 180], dtype=torch.int32)

    # Convert degrees to radians
    horizontal_angles_rad = [
        torch.deg2rad(torch.linspace(-180.0, 180.0 - 360.0 / num_horizontal_angles, num_horizontal_angles))
        for num_horizontal_angles in num_horizontal_angles
    ]

    # create grid
    v_angles = [torch.tensor([0] * x) for x in num_horizontal_angles]
    v_angles = torch.cat(v_angles, dim=0)
    h_angles = torch.cat(horizontal_angles_rad, dim=0)

    # Spherical to Cartesian conversion (assuming Z is up)
    x = torch.cos(v_angles) * torch.cos(h_angles)
    y = torch.cos(v_angles) * torch.sin(h_angles)
    z = torch.sin(v_angles)

    # Ray directions
    ray_directions = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(device)

    # Ray starts: Assuming all rays originate from (0,0,0)
    ray_starts = torch.zeros_like(ray_directions).to(device)
    ray_starts[:, 2] = (
        torch.tensor([[vertical_offsets[i]] * num_horizontal_angles[i] for i in range(len(vertical_offsets))])
        .flatten()
        .to(device)
    )

    # three ray origins
    ray_directions = torch.cat([ray_directions] * 5, dim=0)
    ray_starts = torch.cat(
        [
            ray_starts + torch.tensor([0.0, 0.0, 0.0], device=device),
            ray_starts + torch.tensor([0.3, 0.15, 0.0], device=device),
            ray_starts + torch.tensor([0.3, -0.15, 0.0], device=device),
            ray_starts + torch.tensor([-0.4, 0.15, 0.0], device=device),
            ray_starts + torch.tensor([-0.4, -0.15, 0.0], device=device),
        ],
        dim=0,
    )

    return ray_starts, ray_directions


def ray_3d_3x180_parallel_11x_pattern(cfg: Ray3DPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Lidar sensor pattern for ray casting.

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays.
    """
    # define heights
    vertical_offsets = torch.tensor([0.0, 0.1, 0.2])
    # define angles
    num_horizontal_angles = torch.tensor([180, 180, 180], dtype=torch.int32)

    # Convert degrees to radians
    horizontal_angles_rad = [
        torch.deg2rad(torch.linspace(-180.0, 180.0 - 360.0 / num_horizontal_angles, num_horizontal_angles))
        for num_horizontal_angles in num_horizontal_angles
    ]

    # create grid
    v_angles = [torch.tensor([0] * x) for x in num_horizontal_angles]
    v_angles = torch.cat(v_angles, dim=0)
    h_angles = torch.cat(horizontal_angles_rad, dim=0)

    # Spherical to Cartesian conversion (assuming Z is up)
    x = torch.cos(v_angles) * torch.cos(h_angles)
    y = torch.cos(v_angles) * torch.sin(h_angles)
    z = torch.sin(v_angles)

    # Ray directions
    ray_directions = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(device)

    # Ray starts: Assuming all rays originate from (0,0,0)
    ray_starts = torch.zeros_like(ray_directions).to(device)
    ray_starts[:, 2] = (
        torch.tensor([[vertical_offsets[i]] * num_horizontal_angles[i] for i in range(len(vertical_offsets))])
        .flatten()
        .to(device)
    )

    # three ray origins
    ray_directions = torch.cat([ray_directions] * 11, dim=0)
    ray_starts = torch.cat(
        [
            ray_starts + torch.tensor([0.0, 0.0, 0.0], device=device),
            ray_starts + torch.tensor([0.3, 0.15, 0.0], device=device),
            ray_starts + torch.tensor([0.3, -0.15, 0.0], device=device),
            ray_starts + torch.tensor([-0.4, 0.15, 0.0], device=device),
            ray_starts + torch.tensor([-0.4, -0.15, 0.0], device=device),
            ray_starts + torch.tensor([0.3, 0.0, 0.0], device=device),
            ray_starts + torch.tensor([-0.4, 0.0, 0.0], device=device),
            ray_starts + torch.tensor([0.1, 0.15, 0.0], device=device),
            ray_starts + torch.tensor([0.1, -0.15, 0.0], device=device),
            ray_starts + torch.tensor([-0.2, 0.15, 0.0], device=device),
            ray_starts + torch.tensor([-0.2, -0.15, 0.0], device=device),
        ],
        dim=0,
    )

    return ray_starts, ray_directions


@configclass
class Ray3DPatternCfg(PatternBaseCfg):
    func: Callable = ray_3d_pattern


@configclass
class Ray2DPatternCfg(PatternBaseCfg):
    func: Callable = ray_2d_pattern


@configclass
class Ray3D_45_90_45_30_PattternCfg(PatternBaseCfg):
    func: Callable = ray_3d_45_90_45_30_pattern


@configclass
class Ray3D_4x90_PatternCfg(PatternBaseCfg):
    func: Callable = ray_3d_4x90_pattern


@configclass
class Ray3D_3x180_Parallel_PatternCfg(PatternBaseCfg):
    func: Callable = ray_3d_3x180_parallel_pattern


@configclass
class Ray3D_3x180_Parallel_1x_PatternCfg(PatternBaseCfg):
    func: Callable = ray_3d_3x180_parallel_1x_pattern


@configclass
class Ray3D_3x180_Parallel_3x_PatternCfg(PatternBaseCfg):
    func: Callable = ray_3d_3x180_parallel_3x_pattern


@configclass
class Ray3D_3x180_Parallel_5x_PatternCfg(PatternBaseCfg):
    func: Callable = ray_3d_3x180_parallel_5x_pattern


@configclass
class Ray3D_3x180_Parallel_11x_PatternCfg(PatternBaseCfg):
    func: Callable = ray_3d_3x180_parallel_11x_pattern
