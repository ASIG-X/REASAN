# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom kernels for warp."""

import warp as wp


@wp.kernel(enable_backward=False)
def raycast_mesh_kernel(
    meshes: wp.array(dtype=wp.uint64),
    mesh_pos_buf: wp.array(dtype=wp.vec3),
    mesh_quat_buf: wp.array(dtype=wp.quat),
    num_meshes_per_ray: int,
    ray_starts: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    ray_hits: wp.array(dtype=wp.vec3),
    ray_distance: wp.array(dtype=wp.float32),
    ray_normal: wp.array(dtype=wp.vec3),
    ray_face_id: wp.array(dtype=wp.int32),
    ray_mesh_id: wp.array(dtype=wp.int32),
    max_dist: float = 1e6,
    return_distance: int = False,
    return_normal: int = False,
    return_face_id: int = False,
):
    """Performs ray-casting against a mesh.

    This function performs ray-casting against the given mesh using the provided ray start positions
    and directions. The resulting ray hit positions are stored in the :obj:`ray_hits` array.

    Note that the `ray_starts`, `ray_directions`, and `ray_hits` arrays should have compatible shapes
    and data types to ensure proper execution. Additionally, they all must be in the same frame.

    The function utilizes the `mesh_query_ray` method from the `wp` module to perform the actual ray-casting
    operation. The maximum ray-cast distance is set to `1e6` units.

    Args:
        meshes: The input meshes. The ray-casting is performed against this mesh on the device specified by the
            `mesh`'s `device` attribute. Should have num_meshes_per_ray * N elements.
        mesh_pos_buf, mesh_quat_buf: Same size as meshes. Stores mesh poses. ray_quat_buf should be in xyzw form.
        num_meshes_per_ray: Number of meshes to be tested against for each ray.
        ray_starts: The input ray start positions. Shape is (N, 3).
        ray_directions: The input ray directions. Shape is (N, 3).
        ray_hits: The output ray hit positions. Shape is (N, 3).
        ray_distance: The output ray hit distances. Shape is (N,), if `return_distance` is True. Otherwise,
            this array is not used.
        ray_normal: The output ray hit normals. Shape is (N, 3), if `return_normal` is True. Otherwise,
            this array is not used.
        ray_face_id: The output ray hit face ids. Shape is (N,), if `return_face_id` is True. Otherwise,
            this array is not used.
        ray_mesh_id: The output ray hit mesh index (in the input mesh array). Shape is (N,).
        max_dist: The maximum ray-cast distance. Defaults to 1e6.
        return_distance: Whether to return the ray hit distances. Defaults to False.
        return_normal: Whether to return the ray hit normals. Defaults to False`.
        return_face_id: Whether to return the ray hit face ids. Defaults to False.
    """
    # get the thread id
    tid = wp.tid()

    t = float(0.0)  # hit distance along ray
    u = float(0.0)  # hit face barycentric u
    v = float(0.0)  # hit face barycentric v
    sign = float(0.0)  # hit face sign
    n = wp.vec3()  # hit face normal
    f = int(0)  # hit face index
    source_mesh_id = int(-1)

    hit_success = bool(False)
    for i in range(num_meshes_per_ray):
        idx = tid * num_meshes_per_ray + i
        mesh_id = meshes[idx]

        # compute local ray
        local_ray_start = ray_starts[tid]
        local_ray_direction = ray_directions[tid]
        if i > 0:
            local_ray_start -= mesh_pos_buf[idx]
            local_ray_start = wp.quat_rotate_inv(mesh_quat_buf[idx], local_ray_start)
            local_ray_direction = wp.quat_rotate_inv(mesh_quat_buf[idx], local_ray_direction)

        # ray cast against the mesh and store the hit position.
        succ = wp.mesh_query_ray(mesh_id, local_ray_start, local_ray_direction, max_dist, t, u, v, sign, n, f)
        # decrease max ray distance
        if succ:
            max_dist = wp.min(max_dist, t)
            hit_success = True
            source_mesh_id = int(i)

    # if the ray hit, store the hit data
    if hit_success:
        ray_hits[tid] = ray_starts[tid] + t * ray_directions[tid]
        ray_mesh_id[tid] = source_mesh_id
        if return_distance == 1:
            ray_distance[tid] = t
        if return_normal == 1:
            ray_normal[tid] = n
        if return_face_id == 1:
            ray_face_id[tid] = f
