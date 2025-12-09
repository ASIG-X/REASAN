# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from collections.abc import Sequence
from timeit import default_timer as timer
from typing import Optional

import isaaclab.sim as sim_utils
import numpy as np
import omni.log
import omni.physics.tensors.impl.api as physx
import torch
import warp as wp
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.ray_caster.ray_caster_cfg import RayCasterCfg
from isaaclab.sensors.ray_caster.ray_caster_data import RayCasterData
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_yaw
from isaaclab.utils.warp import convert_to_warp_mesh
from isaacsim.core.prims import XFormPrim
from pxr import UsdGeom, UsdPhysics

from .livox_scan_generator import LivoxScanGenerator
from .raycaster_utils import raycast_mesh


def prim_path_has_regex(prim_path: str):
    return re.match(r"^[a-zA-Z0-9/_]+$", prim_path) is None


class RayCasterDynamic(SensorBase):
    """A ray-casting sensor.

    The ray-caster uses a set of rays to detect collisions with meshes in the scene. The rays are
    defined in the sensor's local coordinate frame. The sensor can be configured to ray-cast against
    a set of meshes with a given ray pattern.

    The meshes are parsed from the list of primitive paths provided in the configuration. These are then
    converted to warp meshes and stored in the `warp_meshes` list. The ray-caster then ray-casts against
    these warp meshes using the ray pattern provided in the configuration.

    .. note::
        Currently, only static meshes are supported. Extending the warp mesh to support dynamic meshes
        is a work in progress.

    .. note::
        This is a modified version of the original RayCaster class in IsaacLab, which allows multiple
        env-specific meshes to be considered. The first prim in mesh_prim_paths should be the terrian,
        which contains no regex. The rest should all be regex paths specifying other prims in each env.
    """

    cfg: RayCasterCfg
    """The configuration parameters."""

    def __init__(self, cfg: RayCasterCfg, sim_mid360: bool):
        """Initializes the ray-caster object.

        Args:
            cfg: The configuration parameters.
        """
        # simulate lidar with isaaclab patterns or mid360 patterns
        self._sim_mid360 = sim_mid360

        # check if sensor path is valid
        # note: currently we do not handle environment indices if there is a regex pattern in the leaf
        #   For example, if the prim path is "/World/Sensor_[1,2]".
        sensor_path = cfg.prim_path.split("/")[-1]
        sensor_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", sensor_path) is None
        if sensor_path_is_regex:
            raise RuntimeError(
                f"Invalid prim path for the ray-caster sensor: {self.cfg.prim_path}."
                "\n\tHint: Please ensure that the prim path does not contain any regex patterns in the leaf."
            )

        # validate mesh_prim_paths
        if prim_path_has_regex(cfg.mesh_prim_paths[0]) or any(
            not prim_path_has_regex(p) for p in cfg.mesh_prim_paths[1:]
        ):
            raise RuntimeError(
                "The first prim in mesh_prim_paths should be the terrian, which contains no regex. "
                "The rest should all be regex paths specifying other prims in each env."
            )

        # Initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data = RayCasterData()
        # the warp meshes used for raycasting.
        self._terrain_mesh: wp.Mesh = None
        self._other_meshes: list[list[wp.Mesh]] = []
        self._other_meshes_prim_views: list = []

        self._ray_mesh_ids: Optional[torch.Tensor] = None

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Ray-caster @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {len(self.meshes)}\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self._view.count

    @property
    def data(self) -> RayCasterData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # resample the drift
        self.drift[env_ids] = self.drift[env_ids].uniform_(*self.cfg.drift_range)

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        # initialize livox scan pattern generator
        if self._sim_mid360:
            self._livox_gen = LivoxScanGenerator(name="mid360", num_envs=self._num_envs, device=self._device)
        # create simulation view
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        # check if the prim at path is an articulated or rigid prim
        # we do this since for physics-based view classes we can access their data directly
        # otherwise we need to use the xform view class which is slower
        found_supported_prim_class = False
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            raise RuntimeError(f"Failed to find a prim at path expression: {self.cfg.prim_path}")
        # create view based on the type of prim
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            self._view = self._physics_sim_view.create_articulation_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            self._view = self._physics_sim_view.create_rigid_body_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        else:
            self._view = XFormPrim(self.cfg.prim_path, reset_xform_properties=False)
            found_supported_prim_class = True
            omni.log.warn(f"The prim at path {prim.GetPath().pathString} is not a physics prim! Using XFormPrim.")
        # check if prim view class is found
        if not found_supported_prim_class:
            raise RuntimeError(f"Failed to find a valid prim view class for the prim paths: {self.cfg.prim_path}")

        # load the meshes by parsing the stage
        self._initialize_warp_meshes()
        # initialize the ray start and directions
        self._initialize_rays_impl()

    def _set_warp_mesh_terrian(self, mesh_prim_path: str):
        # check if the prim is a plane - handle PhysX plane as a special case
        # if a plane exists then we need to create an infinite mesh that is a plane
        mesh_prim = sim_utils.get_first_matching_child_prim(mesh_prim_path, lambda prim: prim.GetTypeName() == "Plane")
        # if we did not find a plane then we need to read the mesh
        if mesh_prim is None:
            # obtain the mesh prim
            mesh_prim = sim_utils.get_first_matching_child_prim(
                mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
            )
            # check if valid
            if mesh_prim is None or not mesh_prim.IsValid():
                raise RuntimeError(f"Invalid mesh prim path: {mesh_prim_path}")
            # cast into UsdGeomMesh
            mesh_prim = UsdGeom.Mesh(mesh_prim)
            # read the vertices and faces
            points = np.asarray(mesh_prim.GetPointsAttr().Get())
            transform_matrix = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
            points = np.matmul(points, transform_matrix[:3, :3].T)
            points += transform_matrix[:3, 3]
            indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
            wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)
            # print info
            omni.log.info(
                f"Read mesh prim: {mesh_prim.GetPath()} with {len(points)} vertices and {len(indices)} faces."
            )
        else:
            mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
            wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
            # print info
            omni.log.info(f"Created infinite plane mesh prim: {mesh_prim.GetPath()}.")
        # add the warp mesh
        self._terrain_mesh = wp_mesh

    def _set_warp_mesh_others(self, mesh_prim_paths: list[str]):
        if not mesh_prim_paths:
            return

        self._other_meshes = [[] for _ in range(self._view.count)]
        self._other_meshes_prim_views: list = []
        for mesh_prim_path in mesh_prim_paths:
            # only handle mesh prims with rigid bodoes
            prim = sim_utils.find_first_matching_prim(mesh_prim_path)
            if prim is None:
                raise RuntimeError(f"Failed to find a prim for mesh_prim_path: {mesh_prim_path}")
            # create view based on the type of prim
            mesh_prim_view = None
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                mesh_prim_view = self._physics_sim_view.create_articulation_view(mesh_prim_path.replace(".*", "*"))
            elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
                mesh_prim_view = self._physics_sim_view.create_rigid_body_view(mesh_prim_path.replace(".*", "*"))
            else:
                mesh_prim_view = XFormPrim(mesh_prim_path, reset_xform_properties=False)
            if mesh_prim_view is None:
                continue
            if mesh_prim_view.count != self._view.count:
                raise RuntimeError(f"invalid mesh prim path: {mesh_prim_path}")

            # store the mesh view
            self._other_meshes_prim_views.append(mesh_prim_view)

            # build warp meshes
            for i, prim_path in enumerate(mesh_prim_view.prim_paths):
                # obtain the mesh prim
                mesh_prim = sim_utils.get_first_matching_child_prim(
                    prim_path, lambda prim: prim.GetTypeName() == "Mesh"
                )
                # check if valid
                if mesh_prim is None or not mesh_prim.IsValid():
                    raise RuntimeError(f"Invalid mesh prim path: {prim_path}")
                # cast into UsdGeomMesh
                mesh_prim = UsdGeom.Mesh(mesh_prim)
                # read the vertices and faces
                points = torch.tensor(mesh_prim.GetPointsAttr().Get())
                indices = torch.tensor(mesh_prim.GetFaceVertexIndicesAttr().Get())
                wp_mesh = convert_to_warp_mesh(points.numpy(), indices.numpy(), device=self.device)
                # print info
                omni.log.info(
                    f"Read mesh prim: {mesh_prim.GetPath()} with {len(points)} vertices and {len(indices)} faces."
                )
                # record the warp_mesh
                self._other_meshes[i].append(wp_mesh)

        if not self._other_meshes[0]:
            self._other_meshes = []

    def _initialize_warp_meshes(self):
        # read prims to ray-cast
        self._set_warp_mesh_terrian(self.cfg.mesh_prim_paths[0])
        self._set_warp_mesh_others(self.cfg.mesh_prim_paths[1:])

        # throw an error if no meshes are found
        if not self._terrain_mesh:
            raise RuntimeError(
                f"No meshes found for ray-casting! Please check the mesh prim paths: {self.cfg.mesh_prim_paths}"
            )

    def _initialize_rays_impl(self):
        # compute ray stars and directions
        if self._sim_mid360:
            self.ray_starts, self.ray_directions = self._livox_gen.sample_rays()
            self.num_rays = self.ray_directions.shape[1]
            # apply offset transformation to the rays
            offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
            offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
            self.ray_directions = quat_apply(offset_quat.repeat(self._num_envs, self.num_rays, 1), self.ray_directions)
            self.ray_starts += offset_pos
            assert self._view.count == self._num_envs
        else:
            self.ray_starts, self.ray_directions = self.cfg.pattern_cfg.func(self.cfg.pattern_cfg, self._device)
            self.num_rays = len(self.ray_directions)
            # apply offset transformation to the rays
            offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
            offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
            self.ray_directions = quat_apply(offset_quat.repeat(len(self.ray_directions), 1), self.ray_directions)
            self.ray_starts += offset_pos
            # repeat the rays for each sensor
            self.ray_starts = self.ray_starts.repeat(self._view.count, 1, 1)
            self.ray_directions = self.ray_directions.repeat(self._view.count, 1, 1)
        # prepare drift
        self.drift = torch.zeros(self._view.count, 3, device=self.device)
        # fill the data buffer
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        self._ray_mesh_ids = torch.full((self._view.count, self.num_rays), -1, device=self.device, dtype=torch.int32)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # update mid360 scan pattern
        if self._sim_mid360:
            self.ray_starts[env_ids], self.ray_directions[env_ids] = self._livox_gen.sample_rays(env_ids=env_ids)
            offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
            offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
            self.ray_directions[env_ids] = quat_apply(
                offset_quat.repeat(len(env_ids), self.num_rays, 1), self.ray_directions[env_ids]
            )
            self.ray_starts += offset_pos

        # obtain the poses of the sensors
        if isinstance(self._view, XFormPrim):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")
        # note: we clone here because we are read-only operations
        pos_w = pos_w.clone()
        quat_w = quat_w.clone()
        # apply drift
        pos_w += self.drift[env_ids]
        # store the poses
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        # update dynamic meshes
        self._other_mesh_pos = torch.zeros(len(env_ids), len(self._other_meshes_prim_views), 3, device=self.device)
        self._other_mesh_quat = torch.zeros(len(env_ids), len(self._other_meshes_prim_views), 4, device=self.device)
        for i_view, mesh_prim_view in enumerate(self._other_meshes_prim_views):
            # retrieve world transformations
            if isinstance(mesh_prim_view, XFormPrim):
                mesh_pos_w, mesh_quat_w = mesh_prim_view.get_world_poses(env_ids)
                mesh_quat_w = convert_quat(mesh_quat_w, to="xyzw")
            elif isinstance(mesh_prim_view, physx.ArticulationView):
                mesh_pos_w, mesh_quat_w = mesh_prim_view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            elif isinstance(mesh_prim_view, physx.RigidBodyView):
                mesh_pos_w, mesh_quat_w = mesh_prim_view.get_transforms()[env_ids].split([3, 4], dim=-1)
            else:
                raise RuntimeError(f"Unsupported mesh view type: {type(mesh_prim_view)}")

            # store mesh poses in all envs
            self._other_mesh_pos[:, i_view, :] = mesh_pos_w
            self._other_mesh_quat[:, i_view, :] = mesh_quat_w

        # ray cast based on the sensor poses
        if self.cfg.attach_yaw_only:
            # only yaw orientation is considered and directions are not rotated
            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            if self.cfg.attach_yaw_only_rotate:
                ray_directions_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
            else:
                ray_directions_w = self.ray_directions[env_ids]
        else:
            # full orientation is considered
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])

        # ray cast and store the hits
        other_meshes = [self._other_meshes[i] for i in env_ids] if self._other_meshes else []
        ray_cast_ret = raycast_mesh(
            ray_starts=ray_starts_w,
            ray_directions=ray_directions_w,
            terrain_mesh=self._terrain_mesh,
            other_meshes=other_meshes,
            other_mesh_pos=self._other_mesh_pos,
            other_mesh_quat=self._other_mesh_quat,
            max_dist=self.cfg.max_distance,
        )
        self._data.ray_hits_w[env_ids] = ray_cast_ret[0]
        self._ray_mesh_ids[env_ids] = ray_cast_ret[-1]

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "ray_visualizer"):
                self.ray_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.ray_visualizer.set_visibility(True)
        else:
            if hasattr(self, "ray_visualizer"):
                self.ray_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # remove possible inf values
        viz_points = self._data.ray_hits_w.reshape(-1, 3)
        viz_points = viz_points[~torch.any(torch.isinf(viz_points), dim=1)]
        # show ray hit positions
        self.ray_visualizer.visualize(viz_points)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._physics_sim_view = None
        self._view = None
