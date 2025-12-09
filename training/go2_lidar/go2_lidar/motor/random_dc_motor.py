from __future__ import annotations

import torch
from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class RandomDCMotor(DCMotor):
    def __init__(self, cfg: RandomDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._motor_strengths = (
            torch.rand(self._num_envs, self.num_joints, device=self._device)
            * (cfg.motor_strengths_range[1] - cfg.motor_strengths_range[0])
            + cfg.motor_strengths_range[0]
        )

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # save current joint vel
        self._joint_vel[:] = joint_vel
        # compute errors
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        # calculate the desired joint torques
        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts
        self.computed_effort *= self._motor_strengths
        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)
        # set the computed actions back into the control action
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class RandomDCMotorCfg(DCMotorCfg):
    """Configuration for a random DC motor actuator model."""

    class_type: type = RandomDCMotor

    motor_strengths_range: tuple[float, float] = (1.0, 1.0)
