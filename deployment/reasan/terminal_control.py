#!/usr/bin/env python3
import argparse
import os
import select
import signal
import sys
import termios
import threading
import time
import tty
from typing import Dict

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node

# Constants
MAX_LINEAR_SPEED = 3.0  # m/s
MAX_LATERAL_SPEED = 2.0  # m/s
MAX_ANGULAR_SPEED = 5.0  # rad/s
SPEED_INCREMENT = 0.05

# Key mappings
KEY_UP = "\x1b[A"
KEY_DOWN = "\x1b[B"
KEY_RIGHT = "\x1b[C"
KEY_LEFT = "\x1b[D"
KEY_A = "a"
KEY_D = "d"
KEY_Q = "q"
KEY_SPACE = " "
KEY_CTRL_C = "\x03"


# For non-blocking key detection
def getch():
    """Gets a single character from standard input, does not echo to the screen."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        # Handle arrow keys (they send multiple chars)
        if ch == "\x1b":
            ch = ch + sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


class KeyboardController(Node):
    def __init__(self, ctrl_mode: str):
        super().__init__("keyboard_controller")

        # Control state
        self.forward_speed = 0.0
        self.lateral_speed = 0.0
        self.yaw_speed = 0.0

        # Create ROS2 publisher for high-level commands
        if ctrl_mode == "loco":
            self.command_publisher = self.create_publisher(Point, "/high_level_command", 10)
        elif ctrl_mode == "filter":
            self.command_publisher = self.create_publisher(Point, "/navigation_vel_cmd", 10)
        else:
            raise ValueError("Invalid control mode. Use 'loco' or 'filter'.")

        # Create command message
        self.command_msg = Point()

        # Initialize command (forward, lateral, angular)
        self.command_msg.x = 0.0  # forward speed
        self.command_msg.y = 0.0  # lateral speed
        self.command_msg.z = 0.0  # angular speed

        # Thread control
        self.running = False
        self.publish_thread = None

        # Initialize to handle terminal resize and exit cleanly
        signal.signal(signal.SIGWINCH, self.handle_resize)
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)

    def handle_resize(self, *args):
        """Handle terminal resize events"""
        # Re-draw the UI
        self.clear_screen()
        self.draw_control_state()

    def handle_interrupt(self, *args):
        """Handle interrupt signals"""
        self.running = False
        self.cleanup()
        sys.exit(0)

    def clear_screen(self):
        """Clear the terminal screen"""
        os.system("clear")

    def draw_control_state(self):
        """Draw the current state to the terminal"""
        self.clear_screen()

        # Draw text instructions and status
        instructions = [
            "Go2 Locomotion Control - Terminal Version",
            "",
            "Controls:",
            "↑/↓: Increase/Decrease Forward Speed",
            "←/→: Increase/Decrease Turn Speed (Left/Right)",
            "a/d: Increase/Decrease Strafe Speed (Left/Right)",
            "SPACE: Clear all commands (STOP)",
            "q: Quit",
            "",
            "=== CURRENT MOVEMENT ===",
        ]

        # Show current movement direction
        movement_status = []
        if self.forward_speed > 0:
            movement_status.append(f"FORWARD ({self.forward_speed:.2f} m/s)")
        elif self.forward_speed < 0:
            movement_status.append(f"BACKWARD ({abs(self.forward_speed):.2f} m/s)")

        if self.yaw_speed > 0:
            movement_status.append(f"TURNING LEFT ({self.yaw_speed:.2f} rad/s)")
        elif self.yaw_speed < 0:
            movement_status.append(f"TURNING RIGHT ({abs(self.yaw_speed):.2f} rad/s)")

        if self.lateral_speed > 0:
            movement_status.append(f"STRAFING LEFT ({self.lateral_speed:.2f} m/s)")
        elif self.lateral_speed < 0:
            movement_status.append(f"STRAFING RIGHT ({abs(self.lateral_speed):.2f} m/s)")

        if not movement_status:
            movement_status.append("STOPPED")

        instructions.extend(movement_status)

        instructions.extend(
            [
                "",
                "=== CURRENT SPEEDS ===",
                f"Forward Speed: {self.forward_speed:.2f} m/s",
                f"Lateral Speed: {self.lateral_speed:.2f} m/s",
                f"Yaw Speed: {self.yaw_speed:.2f} rad/s",
            ]
        )

        # Print all instructions
        print("\n".join(instructions))

    def clear_all_commands(self):
        """Reset all speeds to zero"""
        self.forward_speed = 0.0
        self.lateral_speed = 0.0
        self.yaw_speed = 0.0

    def update_speed_from_key(self, key):
        """Update speeds based on key press with conflict resolution"""
        if key == KEY_UP:  # Forward
            if self.forward_speed <= 0:  # Reset backward movement
                self.forward_speed = 0
            self.forward_speed = min(self.forward_speed + SPEED_INCREMENT, MAX_LINEAR_SPEED)
        elif key == KEY_DOWN:  # Backward
            if self.forward_speed >= 0:  # Reset forward movement
                self.forward_speed = 0
            self.forward_speed = max(self.forward_speed - SPEED_INCREMENT, -MAX_LINEAR_SPEED)
        elif key == KEY_LEFT:  # Turn left
            if self.yaw_speed <= 0:  # Reset right turn
                self.yaw_speed = 0
            self.yaw_speed = min(self.yaw_speed + SPEED_INCREMENT, MAX_ANGULAR_SPEED)
        elif key == KEY_RIGHT:  # Turn right
            if self.yaw_speed >= 0:  # Reset left turn
                self.yaw_speed = 0
            self.yaw_speed = max(self.yaw_speed - SPEED_INCREMENT, -MAX_ANGULAR_SPEED)
        elif key == KEY_A:  # Strafe left
            if self.lateral_speed <= 0:  # Reset right strafe
                self.lateral_speed = 0
            self.lateral_speed = min(self.lateral_speed + SPEED_INCREMENT, MAX_LATERAL_SPEED)
        elif key == KEY_D:  # Strafe right
            if self.lateral_speed >= 0:  # Reset left strafe
                self.lateral_speed = 0
            self.lateral_speed = max(self.lateral_speed - SPEED_INCREMENT, -MAX_LATERAL_SPEED)
        elif key == KEY_SPACE:  # Clear all commands
            self.clear_all_commands()

    def publish_command(self):
        """Publish the current command to the locomotion node"""
        # Set velocities in the command message
        self.command_msg.x = float(self.forward_speed)
        self.command_msg.y = float(self.lateral_speed)
        self.command_msg.z = float(self.yaw_speed)

        # Publish message
        self.command_publisher.publish(self.command_msg)

    def publisher_thread_function(self):
        """Function that runs in the publisher thread"""
        while self.running and rclpy.ok():
            self.publish_command()
            time.sleep(0.02)  # 50 Hz

    def cleanup(self):
        """Clean up before exit"""
        # Send zero commands
        self.forward_speed = 0.0
        self.lateral_speed = 0.0
        self.yaw_speed = 0.0
        self.publish_command()

        # Restore terminal
        os.system("stty sane")
        print("\033[?25h")  # Show cursor

    def run(self):
        """Main control loop"""
        self.running = True

        try:
            # Hide cursor
            print("\033[?25l")

            # Start publisher thread
            self.publish_thread = threading.Thread(target=self.publisher_thread_function)
            self.publish_thread.daemon = True
            self.publish_thread.start()

            # Draw the initial UI
            self.draw_control_state()

            # Main input loop - process one key at a time
            while self.running and rclpy.ok():
                key = getch()

                # Process key
                if key == KEY_Q or key == KEY_CTRL_C:  # q or Ctrl+C
                    self.running = False
                    break

                # Update speed based on key press
                if key in [KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_A, KEY_D, KEY_SPACE]:
                    self.update_speed_from_key(key)
                    self.draw_control_state()

        except KeyboardInterrupt:
            self.running = False
            print("Keyboard interrupt received, shutting down")
        finally:
            self.cleanup()

            if self.publish_thread:
                self.publish_thread.join(timeout=1.0)

            print("Exiting controller")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Go2 Locomotion Control")
    parser.add_argument(
        "--control",
        type=str,
        choices=["loco", "filter"],
        default="loco",
        help="Control mode: 'loco' for high-level commands, 'filter' for velocity commands",
    )
    args = parser.parse_args()

    # Initialize ROS2
    rclpy.init()

    controller = KeyboardController(ctrl_mode=args.control)

    # Spin in a separate thread to handle ROS2 callbacks
    ros_thread = threading.Thread(target=lambda: rclpy.spin(controller))
    ros_thread.daemon = True
    ros_thread.start()

    try:
        controller.run()
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down")
    finally:
        rclpy.shutdown()
        print("Exiting controller")
