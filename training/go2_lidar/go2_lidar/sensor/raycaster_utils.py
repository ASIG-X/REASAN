# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapping around warp kernels for compatibility with torch tensors."""

# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

from timeit import default_timer as timer

import torch
import warp as wp

# disable warp module initialization messages
wp.config.quiet = True
# initialize the warp module
wp.init()

from .raycaster_kernels import raycast_mesh_kernel


def raycast_mesh(
    ray_starts: torch.Tensor,
    ray_directions: torch.Tensor,
    terrain_mesh: wp.Mesh,
    other_meshes: list[list[wp.Mesh]] = [],
    other_mesh_pos: torch.Tensor = None,
    other_mesh_quat: torch.Tensor = None,
    max_dist: float = 1e6,
    return_distance: bool = False,
    return_normal: bool = False,
    return_face_id: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Performs ray-casting against a mesh.

    Note that the `ray_starts` and `ray_directions`, and `ray_hits` should have compatible shapes
    and data types to ensure proper execution. Additionally, they all must be in the same frame.

    Args:
        ray_starts: The starting position of the rays. Shape (N, 3).
        ray_directions: The ray directions for each ray. Shape (N, 3).
        terrain_mesh: The warp mesh for the terrain to ray-cast against. This is shared by all envs.
        other_meshes: Other meshes in each env to consider. The sub-lists should have equal sizes.
        other_mesh_pos: Positions of other meshes. Shape (num_envs, num_meshes_per_env, 3)
        other_mesh_quat: Orientations of other meshes. *Quaternion in xyzw form, which is different from IsaacLab.*
        max_dist: The maximum distance to ray-cast. Defaults to 1e6.
        return_distance: Whether to return the distance of the ray until it hits the mesh. Defaults to False.
        return_normal: Whether to return the normal of the mesh face the ray hits. Defaults to False.
        return_face_id: Whether to return the face id of the mesh face the ray hits. Defaults to False.

    Returns:
        The ray hit position. Shape (N, 3).
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit distance. Shape (N,).
            Will only return if :attr:`return_distance` is True, else returns None.
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit normal. Shape (N, 3).
            Will only return if :attr:`return_normal` is True else returns None.
            The returned tensor contains :obj:`float('inf')` for missed hits.
        The ray hit face id. Shape (N,).
            Will only return if :attr:`return_face_id` is True else returns None.
            The returned tensor contains :obj:`int(-1)` for missed hits.
    """
    # extract device and shape information
    shape = ray_starts.shape
    device = ray_starts.device

    # validate other_meshes
    num_meshes_per_env = 1
    if other_meshes:
        # make sure we have the same number of meshes for each env
        num_other_meshes = set(len(x) for x in other_meshes)
        if len(num_other_meshes) != 1:
            raise RuntimeError("Each env must have exactly the same number of meshes; please check other_meshes.")
        # make sure we have at least one mesh in other_meshes if it is specified
        num_other_meshes = next(iter(num_other_meshes))
        if num_other_meshes == 0:
            raise RuntimeError(
                "Please specify at least one mesh for each env in other_meshes; or you can just omit this argument."
            )
        # make sure the number of envs matches
        if len(other_meshes) != shape[0]:
            raise RuntimeError("Dimensions of other_meshes and ray_starts does not match.")
        # make sure the mesh poses are given
        if other_mesh_pos is None or other_mesh_quat is None:
            raise RuntimeError("other_mesh_pos and other_mesh_quat must be specified if other_meshes if given.")
        # make sure the mesh poses have correct shape
        if (
            other_mesh_pos.shape[:2] != other_mesh_quat.shape[:2]
            or other_mesh_pos.shape[2] != 3
            or other_mesh_quat.shape[2] != 4
            or other_mesh_pos.shape[0] != len(other_meshes)
            or other_mesh_pos.shape[1] != num_other_meshes
        ):
            raise RuntimeError("other_mesh_pos or other_mesh_quat have wrong shape.")
        num_meshes_per_env += num_other_meshes

    # device of the mesh
    torch_device = wp.device_to_torch(terrain_mesh.device)
    # reshape the tensors
    ray_starts = ray_starts.to(torch_device).view(-1, 3).contiguous()
    ray_directions = ray_directions.to(torch_device).view(-1, 3).contiguous()
    num_rays = ray_starts.shape[0]
    # create output tensor for the ray hits
    ray_hits = torch.full((num_rays, 3), float("inf"), device=torch_device).contiguous()

    # create mesh id tensor
    mesh_ids = torch.zeros(shape[0], shape[1], num_meshes_per_env, device=torch_device, dtype=torch.int64)
    mesh_ids[:, :, 0] = terrain_mesh.id
    if other_meshes:
        mesh_ids[:, :, 1:] = torch.tensor([[[m.id for m in meshes]] for meshes in other_meshes])
    mesh_ids = mesh_ids.flatten().contiguous()

    # create buffers for mesh poses for each ray
    ray_mesh_pos_buf = torch.zeros(shape[0], shape[1], num_meshes_per_env, 3, device=torch_device)
    ray_mesh_pos_buf[:, :, 0, :] = 0
    ray_mesh_quat_buf = torch.zeros(shape[0], shape[1], num_meshes_per_env, 4, device=torch_device)
    ray_mesh_quat_buf[:, :, 0, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=torch_device)
    if other_meshes:
        ray_mesh_pos_buf[:, :, 1:, :] = other_mesh_pos.unsqueeze(dim=1)
        ray_mesh_quat_buf[:, :, 1:, :] = other_mesh_quat.unsqueeze(dim=1)
    ray_mesh_pos_buf = ray_mesh_pos_buf.view(-1, 3).contiguous()
    ray_mesh_quat_buf = ray_mesh_quat_buf.view(-1, 4).contiguous()

    # map the memory to warp arrays
    ray_starts_wp = wp.from_torch(ray_starts, dtype=wp.vec3)
    ray_directions_wp = wp.from_torch(ray_directions, dtype=wp.vec3)
    ray_hits_wp = wp.from_torch(ray_hits, dtype=wp.vec3)
    mesh_ids_wp = wp.from_torch(mesh_ids, dtype=wp.uint64)
    ray_mesh_pos_buf_wp = wp.from_torch(ray_mesh_pos_buf, dtype=wp.vec3)
    ray_mesh_quat_buf_wp = wp.from_torch(ray_mesh_quat_buf, dtype=wp.quat)

    ray_mesh_id_buf = torch.ones(num_rays, dtype=torch.int32, device=torch_device).contiguous() * (-1)
    ray_mesh_id_wp = wp.from_torch(ray_mesh_id_buf, dtype=wp.int32)

    if return_distance:
        ray_distance = torch.full((num_rays,), float("inf"), device=torch_device).contiguous()
        ray_distance_wp = wp.from_torch(ray_distance, dtype=wp.float32)
    else:
        ray_distance = None
        ray_distance_wp = wp.empty((1,), dtype=wp.float32, device=torch_device)

    if return_normal:
        ray_normal = torch.full((num_rays, 3), float("inf"), device=torch_device).contiguous()
        ray_normal_wp = wp.from_torch(ray_normal, dtype=wp.vec3)
    else:
        ray_normal = None
        ray_normal_wp = wp.empty((1,), dtype=wp.vec3, device=torch_device)

    if return_face_id:
        ray_face_id = torch.ones((num_rays,), dtype=torch.int32, device=torch_device).contiguous() * (-1)
        ray_face_id_wp = wp.from_torch(ray_face_id, dtype=wp.int32)
    else:
        ray_face_id = None
        ray_face_id_wp = wp.empty((1,), dtype=wp.int32, device=torch_device)

    # launch the warp kernel
    wp.launch(
        kernel=raycast_mesh_kernel,
        dim=num_rays,
        inputs=[
            mesh_ids_wp,
            ray_mesh_pos_buf_wp,
            ray_mesh_quat_buf_wp,
            num_meshes_per_env,
            ray_starts_wp,
            ray_directions_wp,
            ray_hits_wp,
            ray_distance_wp,
            ray_normal_wp,
            ray_face_id_wp,
            ray_mesh_id_wp,
            float(max_dist),
            int(return_distance),
            int(return_normal),
            int(return_face_id),
        ],
        device=terrain_mesh.device,
    )
    # NOTE: Synchronize is not needed anymore, but we keep it for now. Check with @dhoeller.
    wp.synchronize()

    if return_distance:
        ray_distance = ray_distance.to(device).view(shape[0], shape[1])
    if return_normal:
        ray_normal = ray_normal.to(device).view(shape)
    if return_face_id:
        ray_face_id = ray_face_id.to(device).view(shape[0], shape[1])

    return (
        ray_hits.to(device).view(shape),
        ray_distance,
        ray_normal,
        ray_face_id,
        ray_mesh_id_buf.to(device).view(shape[0], shape[1]),
    )
