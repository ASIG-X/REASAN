import argparse
import struct
import sys
import time

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber

# Uncomment the following two lines when using Go2、Go2-W、B2、B2-W、H1 robot
# from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
# from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
# Uncomment the following two lines when using G1、H1-2 robot
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

# Constants
MAX_LINEAR_SPEED = 2.5  # m/s
MAX_LATERAL_SPEED = 1.5  # m/s
MAX_ANGULAR_SPEED = 3.0  # rad/s


class unitreeRemoteController:
    def __init__(self):
        # Analog stick values
        self.Lx = 0.0  # Left stick X (strafe)
        self.Ly = 0.0  # Left stick Y (forward/backward)
        self.Rx = 0.0  # Right stick X (turn)
        self.Ry = 0.0  # Right stick Y (unused)

        # Button states
        self.L1 = 0
        self.L2 = 0
        self.R1 = 0
        self.R2 = 0
        self.A = 0
        self.B = 0
        self.X = 0
        self.Y = 0
        self.Up = 0
        self.Down = 0
        self.Left = 0
        self.Right = 0
        self.Select = 0
        self.F1 = 0
        self.F3 = 0
        self.Start = 0

    def parse_botton(self, data1, data2):
        self.R1 = (data1 >> 0) & 1
        self.L1 = (data1 >> 1) & 1
        self.Start = (data1 >> 2) & 1
        self.Select = (data1 >> 3) & 1
        self.R2 = (data1 >> 4) & 1
        self.L2 = (data1 >> 5) & 1
        self.F1 = (data1 >> 6) & 1
        self.F3 = (data1 >> 7) & 1
        self.A = (data2 >> 0) & 1
        self.B = (data2 >> 1) & 1
        self.X = (data2 >> 2) & 1
        self.Y = (data2 >> 3) & 1
        self.Up = (data2 >> 4) & 1
        self.Right = (data2 >> 5) & 1
        self.Down = (data2 >> 6) & 1
        self.Left = (data2 >> 7) & 1

    def parse_key(self, data):
        lx_offset = 4
        self.Lx = struct.unpack("<f", data[lx_offset : lx_offset + 4])[0]
        rx_offset = 8
        self.Rx = struct.unpack("<f", data[rx_offset : rx_offset + 4])[0]
        ry_offset = 12
        self.Ry = struct.unpack("<f", data[ry_offset : ry_offset + 4])[0]
        ly_offset = 20
        self.Ly = struct.unpack("<f", data[ly_offset : ly_offset + 4])[0]

    def parse(self, remoteData):
        self.parse_key(remoteData)
        self.parse_botton(remoteData[2], remoteData[3])


class WirelessController(Node):
    def __init__(self, ctrl_mode: str):
        super().__init__("wireless_controller")

        self.low_state = None
        self.remoteController = unitreeRemoteController()

        # Create ROS2 publisher based on control mode
        if ctrl_mode == "loco":
            self.command_publisher = self.create_publisher(Point, "/high_level_command", 10)
        elif ctrl_mode == "filter":
            self.command_publisher = self.create_publisher(Point, "/navigation_vel_cmd", 10)
        else:
            raise ValueError("Invalid control mode. Use 'loco' or 'filter'.")

        # Create command message
        self.command_msg = Point()

        # Timer for publishing commands at regular intervals
        self.timer = self.create_timer(0.02, self.publish_command)  # 50 Hz

        # Deadzone for analog sticks
        self.deadzone = 0.1

        # Smoothed values for gradual ramp-up (but instant zero on release)
        self.smoothed_forward = 0.0
        self.smoothed_lateral = 0.0
        self.smoothed_angular = 0.0
        self.smooth_factor = 0.5  # Higher = more responsive

    def Init(self):
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

    def apply_deadzone(self, value, deadzone=None):
        """Apply deadzone to analog input"""
        if deadzone is None:
            deadzone = self.deadzone
        if abs(value) < deadzone:
            return 0.0
        return value

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg
        wireless_remote_data = self.low_state.wireless_remote
        self.remoteController.parse(wireless_remote_data)

    def publish_command(self):
        """Publish movement command based on controller input"""
        if self.low_state is None:
            return

        # Apply deadzone to stick inputs
        forward_input = self.apply_deadzone(self.remoteController.Ly)  # Negative for forward
        lateral_input = self.apply_deadzone(-self.remoteController.Lx)  # Left stick X for strafe
        angular_input = self.apply_deadzone(-self.remoteController.Rx)  # Negative for left turn

        # Crispy control: instant zero on release, smooth ramp-up otherwise
        # If input is zero (stick released), immediately set to zero
        # Otherwise, apply smoothing for gradual acceleration
        if forward_input == 0.0:
            self.smoothed_forward = 0.0
        else:
            self.smoothed_forward = (
                self.smoothed_forward * (1 - self.smooth_factor) + forward_input * self.smooth_factor
            )

        if lateral_input == 0.0:
            self.smoothed_lateral = 0.0
        else:
            self.smoothed_lateral = (
                self.smoothed_lateral * (1 - self.smooth_factor) + lateral_input * self.smooth_factor
            )

        if angular_input == 0.0:
            self.smoothed_angular = 0.0
        else:
            self.smoothed_angular = (
                self.smoothed_angular * (1 - self.smooth_factor) + angular_input * self.smooth_factor
            )

        # Map smoothed inputs to velocities
        self.command_msg.x = float(self.smoothed_forward * MAX_LINEAR_SPEED)
        self.command_msg.y = float(self.smoothed_lateral * MAX_LATERAL_SPEED)
        self.command_msg.z = float(self.smoothed_angular * MAX_ANGULAR_SPEED)

        # Publish the command
        self.command_publisher.publish(self.command_msg)

        # Optional: Print current velocities (can be removed for production)
        print(
            f"Velocities - Forward: {self.command_msg.x:.2f}, "
            f"Lateral: {self.command_msg.y:.2f}, "
            f"Angular: {self.command_msg.z:.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Go2 Wireless Controller")
    parser.add_argument("--net", type=str, default="lo", help="Network interface for wireless communication")
    parser.add_argument(
        "--control",
        type=str,
        choices=["loco", "filter"],
        default="loco",
        help="Control mode: 'loco' for high-level commands, 'filter' for velocity commands",
    )
    args = parser.parse_args()

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    print("Controller mapping:")
    print("- Left stick: Forward/Backward (Y-axis) and Strafe Left/Right (X-axis)")
    print("- Right stick: Turn Left/Right (X-axis)")

    # Initialize Unitree SDK
    print(f"Using network interface: {args.net}")
    ChannelFactoryInitialize(0, args.net)

    # Initialize ROS2
    rclpy.init()

    # Create controller instance
    controller = WirelessController(ctrl_mode=args.control)
    controller.Init()

    try:
        # Spin ROS2 node
        rclpy.spin(controller)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down")
    finally:
        # Send zero commands before shutdown
        controller.command_msg.x = 0.0
        controller.command_msg.y = 0.0
        controller.command_msg.z = 0.0
        controller.command_publisher.publish(controller.command_msg)

        rclpy.shutdown()
        print("Wireless controller shutdown complete")
