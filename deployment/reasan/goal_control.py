#!/usr/bin/env python3

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pygame
import rclpy
from geometry_msgs.msg import Point
from livox_ros_driver2.msg import CustomMsg
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool


@dataclass
class LidarFrame:
    """Store raw lidar data before transformation"""

    points: np.ndarray  # Nx3 array in body frame
    timestamp: float


class GoalControlNode(Node):
    def __init__(self):
        super().__init__("goal_control_node")

        # Robot state
        self.robot_pos = np.array([0.0, 0.0, 0.0])
        self.robot_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z

        # Goal/waypoint system
        self.waypoints: List[np.ndarray] = []  # List of waypoint positions
        self.current_waypoint_index = 0
        self.goal_pos = np.array([0.0, 0.0])
        self.goal_reached_threshold = 1.0  # meters

        # Lidar buffer - keep recent frames
        self.lidar_buffer: deque[LidarFrame] = deque(maxlen=50)

        # Current point cloud (transformed to world frame, synchronized with odometry)
        self.current_pointcloud: Optional[np.ndarray] = None

        # Track last used lidar timestamp to avoid re-transforming same frame
        self.last_used_lidar_timestamp: Optional[float] = None

        # Z-filtering thresholds (default 0-2m)
        self.z_min = 0.0  # meters
        self.z_max = 2.0  # meters

        # Subscriptions
        self.odom_sub = self.create_subscription(Odometry, "/go2_odom", self.odom_callback, 10)

        self.lidar_sub = self.create_subscription(CustomMsg, "/livox/lidar", self.lidar_callback, 10)

        # Publisher
        self.goal_pub = self.create_publisher(Point, "/navigation_goal", 10)
        self.goal_reached_pub = self.create_publisher(Bool, "/goal_reached", 10)

        # Timer to publish goal at 10Hz
        self.goal_timer = self.create_timer(0.1, self.publish_goal)

        self.get_logger().info("Goal Control Node initialized")

    def odom_callback(self, msg: Odometry):
        """Update robot position and orientation, then transform closest lidar frame"""
        self.robot_pos[0] = msg.pose.pose.position.x
        self.robot_pos[1] = msg.pose.pose.position.y
        self.robot_pos[2] = msg.pose.pose.position.z

        self.robot_quat[0] = msg.pose.pose.orientation.w
        self.robot_quat[1] = msg.pose.pose.orientation.x
        self.robot_quat[2] = msg.pose.pose.orientation.y
        self.robot_quat[3] = msg.pose.pose.orientation.z

        # Check if current goal is reached and advance to next waypoint
        self.check_goal_reached()

        # Get odometry timestamp
        odom_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Find lidar frame closest in time to this odometry message
        if self.lidar_buffer:
            closest_frame = min(self.lidar_buffer, key=lambda f: abs(f.timestamp - odom_time))
            time_diff = abs(closest_frame.timestamp - odom_time)

            # Only use if:
            # 1. Reasonably close in time (within 50ms for better sync)
            # 2. Haven't already used this exact frame
            if time_diff < 0.05 and closest_frame.timestamp != self.last_used_lidar_timestamp:
                self.current_pointcloud = self.transform_to_world(closest_frame.points)
                self.last_used_lidar_timestamp = closest_frame.timestamp

                # Log sync quality for debugging
                if time_diff > 0.02:
                    self.get_logger().debug(f"Sync offset: {time_diff * 1000:.1f}ms")

            # Remove old lidar frames (older than 0.3 seconds for tighter buffer)
            cutoff_time = odom_time - 0.3
            while self.lidar_buffer and self.lidar_buffer[0].timestamp < cutoff_time:
                self.lidar_buffer.popleft()

    def lidar_callback(self, msg: CustomMsg):
        """Buffer LiDAR frames without transforming"""
        if msg.point_num == 0:
            return

        # Extract points from CustomMsg
        points_body = np.zeros((msg.point_num, 3), dtype=np.float32)
        for i, point in enumerate(msg.points):
            points_body[i] = [point.x, point.y, point.z]

        # Filter out zero points
        valid_mask = np.linalg.norm(points_body, axis=1) > 1e-3
        points_body = points_body[valid_mask]

        if len(points_body) == 0:
            return

        # Get lidar timestamp
        lidar_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Add to buffer (do NOT transform yet - wait for matching odometry)
        frame = LidarFrame(points=points_body, timestamp=lidar_time)
        self.lidar_buffer.append(frame)

    def transform_to_world(self, points_body: np.ndarray) -> np.ndarray:
        """Transform points from body frame to world frame"""
        # Create rotation matrix from quaternion
        w, x, y, z = self.robot_quat
        rot_matrix = np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
            ]
        )

        # Rotate and translate
        points_world = points_body @ rot_matrix.T + self.robot_pos
        return points_world

    def publish_goal(self):
        """Publish current goal"""
        msg = Point()
        msg.x = float(self.goal_pos[0])
        msg.y = float(self.goal_pos[1])
        msg.z = 0.0
        self.goal_pub.publish(msg)

    def check_goal_reached(self):
        """Check if robot has reached current goal and advance to next waypoint"""
        if len(self.waypoints) == 0:
            return

        # Calculate distance to current goal
        dist = np.linalg.norm(self.robot_pos[:2] - self.goal_pos)

        if dist < self.goal_reached_threshold:
            # Publish goal reached message
            goal_reached_msg = Bool()
            goal_reached_msg.data = True
            self.goal_reached_pub.publish(goal_reached_msg)

            # Goal reached, advance to next waypoint
            self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)
            self.goal_pos = self.waypoints[self.current_waypoint_index].copy()
            self.get_logger().info(
                f"Goal reached! Moving to waypoint {self.current_waypoint_index + 1}/{len(self.waypoints)}: "
                f"({self.goal_pos[0]:.2f}, {self.goal_pos[1]:.2f})"
            )

    def set_goal(self, x: float, y: float):
        """Set a single goal (clears waypoints and sets one waypoint)"""
        self.waypoints = [np.array([x, y])]
        self.current_waypoint_index = 0
        self.goal_pos = self.waypoints[0].copy()
        self.get_logger().info(f"Single goal set: ({x:.2f}, {y:.2f})")

    def add_waypoint(self, x: float, y: float):
        """Add a waypoint to the list"""
        self.waypoints.append(np.array([x, y]))
        # If this is the first waypoint, set it as current goal
        if len(self.waypoints) == 1:
            self.current_waypoint_index = 0
            self.goal_pos = self.waypoints[0].copy()
        self.get_logger().info(f"Waypoint {len(self.waypoints)} added: ({x:.2f}, {y:.2f})")

    def clear_waypoints(self):
        """Clear all waypoints"""
        self.waypoints = []
        self.current_waypoint_index = 0
        self.get_logger().info("All waypoints cleared")

    def set_waypoints(self, waypoints: List[Tuple[float, float]]):
        """Set multiple waypoints at once"""
        self.waypoints = [np.array([x, y]) for x, y in waypoints]
        if len(self.waypoints) > 0:
            self.current_waypoint_index = 0
            self.goal_pos = self.waypoints[0].copy()
            self.get_logger().info(f"Set {len(self.waypoints)} waypoints")
        else:
            self.get_logger().info("No waypoints set")


class GoalControlVisualizer:
    def __init__(self, node: GoalControlNode, width: int = 1280, height: int = 800):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Goal Control - Top-Down View")
        self.clock = pygame.time.Clock()

        self.node = node
        self.width = width
        self.height = height

        # View parameters
        self.scale = 50.0  # pixels per meter
        self.offset_x = width / 2
        self.offset_y = height / 2
        self.view_rotation = 0.0  # radians

        # Interaction state
        self.dragging = False
        self.last_mouse_pos = None
        self.panning = False
        self.dragging_z_min_slider = False
        self.dragging_z_max_slider = False
        self.waypoint_mode = False  # Shift+Click to add waypoints

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_ROBOT = (0, 255, 0)
        self.COLOR_GOAL = (255, 0, 0)
        self.COLOR_LIDAR = (180, 180, 180)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_BUTTON = (70, 70, 90)
        self.COLOR_BUTTON_HOVER = (90, 90, 120)
        self.COLOR_BUTTON_TEXT = (255, 255, 255)
        self.COLOR_SLIDER_BG = (50, 50, 60)
        self.COLOR_SLIDER_FILL = (80, 120, 200)
        self.COLOR_SLIDER_HANDLE = (150, 180, 255)

        self.font = pygame.font.SysFont("monospace", 14)
        self.button_font = pygame.font.SysFont("monospace", 16, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 12)

        # Reset button
        self.reset_button_rect = pygame.Rect(self.width - 180, 10, 170, 40)
        self.reset_button_hovered = False

        # Clear waypoints button
        self.clear_button_rect = pygame.Rect(self.width - 180, 60, 170, 40)
        self.clear_button_hovered = False

        # Z-filter sliders (bottom of screen)
        slider_width = 300
        slider_height = 20
        slider_margin = 20
        slider_y = self.height - 100

        self.z_min_slider_rect = pygame.Rect(slider_margin, slider_y, slider_width, slider_height)
        self.z_max_slider_rect = pygame.Rect(slider_margin, slider_y + 40, slider_width, slider_height)

        # Z range for sliders
        self.z_slider_min = -3.0
        self.z_slider_max = 3.0

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        # Apply view rotation
        cos_r = math.cos(self.view_rotation)
        sin_r = math.sin(self.view_rotation)
        x_rot = x * cos_r - y * sin_r
        y_rot = x * sin_r + y * cos_r

        # Scale and offset
        screen_x = int(x_rot * self.scale + self.offset_x)
        screen_y = int(-y_rot * self.scale + self.offset_y)  # Flip Y for screen coords
        return screen_x, screen_y

    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates"""
        # Remove offset and scale
        x_rot = (screen_x - self.offset_x) / self.scale
        y_rot = -(screen_y - self.offset_y) / self.scale  # Flip Y back

        # Apply inverse rotation
        cos_r = math.cos(-self.view_rotation)
        sin_r = math.sin(-self.view_rotation)
        x = x_rot * cos_r - y_rot * sin_r
        y = x_rot * sin_r + y_rot * cos_r

        return x, y

    def draw_grid(self):
        """Draw grid lines"""
        grid_spacing = 1.0  # meters

        # Determine visible range
        x_min, y_max = self.screen_to_world(0, 0)
        x_max, y_min = self.screen_to_world(self.width, self.height)

        # Draw vertical lines
        x_start = math.floor(x_min / grid_spacing) * grid_spacing
        x_end = math.ceil(x_max / grid_spacing) * grid_spacing
        x = x_start
        while x <= x_end:
            sx1, sy1 = self.world_to_screen(x, y_min)
            sx2, sy2 = self.world_to_screen(x, y_max)
            if 0 <= sx1 <= self.width or 0 <= sx2 <= self.width:
                pygame.draw.line(self.screen, self.COLOR_GRID, (sx1, sy1), (sx2, sy2), 1)
            x += grid_spacing

        # Draw horizontal lines
        y_start = math.floor(y_min / grid_spacing) * grid_spacing
        y_end = math.ceil(y_max / grid_spacing) * grid_spacing
        y = y_start
        while y <= y_end:
            sx1, sy1 = self.world_to_screen(x_min, y)
            sx2, sy2 = self.world_to_screen(x_max, y)
            if 0 <= sy1 <= self.height or 0 <= sy2 <= self.height:
                pygame.draw.line(self.screen, self.COLOR_GRID, (sx1, sy1), (sx2, sy2), 1)
            y += grid_spacing

        # Draw origin axes
        origin_sx, origin_sy = self.world_to_screen(0, 0)
        x_axis_sx, x_axis_sy = self.world_to_screen(1, 0)
        y_axis_sx, y_axis_sy = self.world_to_screen(0, 1)

        pygame.draw.line(
            self.screen, (100, 100, 255), (origin_sx, origin_sy), (x_axis_sx, x_axis_sy), 2
        )  # X-axis (blue)
        pygame.draw.line(
            self.screen, (100, 255, 100), (origin_sx, origin_sy), (y_axis_sx, y_axis_sy), 2
        )  # Y-axis (green)

    def draw_point_cloud(self):
        """Draw current point cloud with Z-filtering"""
        if self.node.current_pointcloud is None:
            return

        # Filter points by Z coordinate
        z_filtered = self.node.current_pointcloud[
            (self.node.current_pointcloud[:, 2] >= self.node.z_min)
            & (self.node.current_pointcloud[:, 2] <= self.node.z_max)
        ]

        # Draw points (larger size)
        for point in z_filtered[::5]:  # Subsample for performance
            sx, sy = self.world_to_screen(point[0], point[1])
            if 0 <= sx < self.width and 0 <= sy < self.height:
                pygame.draw.circle(self.screen, self.COLOR_LIDAR, (sx, sy), 2)

    def draw_robot(self):
        """Draw robot as arrow"""
        # Robot position
        rx, ry, _ = self.node.robot_pos
        sx, sy = self.world_to_screen(rx, ry)

        # Calculate robot heading
        w, x, y, z = self.node.robot_quat
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        # Arrow points
        length = 0.5  # meters
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        front_x = rx + length * cos_yaw
        front_y = ry + length * sin_yaw
        left_x = rx + 0.2 * cos_yaw - 0.2 * sin_yaw
        left_y = ry + 0.2 * sin_yaw + 0.2 * cos_yaw
        right_x = rx + 0.2 * cos_yaw + 0.2 * sin_yaw
        right_y = ry + 0.2 * sin_yaw - 0.2 * cos_yaw

        front_sx, front_sy = self.world_to_screen(front_x, front_y)
        left_sx, left_sy = self.world_to_screen(left_x, left_y)
        right_sx, right_sy = self.world_to_screen(right_x, right_y)

        # Draw arrow
        pygame.draw.polygon(
            self.screen, self.COLOR_ROBOT, [(front_sx, front_sy), (left_sx, left_sy), (right_sx, right_sy)]
        )
        pygame.draw.circle(self.screen, self.COLOR_ROBOT, (sx, sy), 8, 2)

    def draw_goal(self):
        """Draw current goal and all waypoints"""
        # Draw all waypoints
        for i, waypoint in enumerate(self.node.waypoints):
            wx, wy = waypoint
            sx, sy = self.world_to_screen(wx, wy)

            # Determine color (current goal is brighter)
            if i == self.node.current_waypoint_index:
                color = self.COLOR_GOAL
                size = 15
            else:
                color = (150, 0, 0)  # Darker red for inactive waypoints
                size = 12

            # Draw crosshair
            pygame.draw.line(self.screen, color, (sx - size, sy), (sx + size, sy), 3)
            pygame.draw.line(self.screen, color, (sx, sy - size), (sx, sy + size), 3)
            pygame.draw.circle(self.screen, color, (sx, sy), 10, 2)

            # Draw waypoint number
            number_text = str(i + 1)
            number_surface = self.small_font.render(number_text, True, color)
            self.screen.blit(number_surface, (sx + 15, sy - 15))

        # Draw lines connecting waypoints if more than one
        if len(self.node.waypoints) > 1:
            for i in range(len(self.node.waypoints)):
                start_wp = self.node.waypoints[i]
                end_wp = self.node.waypoints[(i + 1) % len(self.node.waypoints)]
                start_sx, start_sy = self.world_to_screen(start_wp[0], start_wp[1])
                end_sx, end_sy = self.world_to_screen(end_wp[0], end_wp[1])
                pygame.draw.line(self.screen, (100, 50, 50), (start_sx, start_sy), (end_sx, end_sy), 1)

    def draw_slider(self, rect: pygame.Rect, value: float, min_val: float, max_val: float, label: str):
        """Draw a slider control"""
        # Background
        pygame.draw.rect(self.screen, self.COLOR_SLIDER_BG, rect, border_radius=3)

        # Fill (from min to current value)
        norm_value = (value - min_val) / (max_val - min_val)
        fill_width = int(rect.width * norm_value)
        fill_rect = pygame.Rect(rect.x, rect.y, fill_width, rect.height)
        pygame.draw.rect(self.screen, self.COLOR_SLIDER_FILL, fill_rect, border_radius=3)

        # Handle
        handle_x = rect.x + fill_width
        handle_rect = pygame.Rect(handle_x - 5, rect.y - 3, 10, rect.height + 6)
        pygame.draw.rect(self.screen, self.COLOR_SLIDER_HANDLE, handle_rect, border_radius=3)

        # Border
        pygame.draw.rect(self.screen, self.COLOR_TEXT, rect, 1, border_radius=3)

        # Label
        label_text = f"{label}: {value:.2f}m"
        label_surface = self.small_font.render(label_text, True, self.COLOR_TEXT)
        self.screen.blit(label_surface, (rect.x, rect.y - 18))

    def draw_ui(self):
        """Draw UI overlay"""
        texts = [
            f"Robot: ({self.node.robot_pos[0]:.2f}, {self.node.robot_pos[1]:.2f})",
            f"Goal: ({self.node.goal_pos[0]:.2f}, {self.node.goal_pos[1]:.2f})",
            f"Waypoints: {self.node.current_waypoint_index + 1}/{len(self.node.waypoints)}"
            if self.node.waypoints
            else "Waypoints: 0/0",
            f"Scale: {self.scale:.1f} px/m",
            f"Rotation: {math.degrees(self.view_rotation):.1f}°",
            "",
            "Controls:",
            "Left-click: Set goal",
            "Shift+Click: Add waypoint",
            "Middle-drag: Pan view",
            "Right-drag: Rotate view",
            "Mouse wheel: Zoom",
            "R: Reset view",
            "ESC: Quit",
        ]

        y_offset = 10
        for text in texts:
            surface = self.font.render(text, True, self.COLOR_TEXT)
            self.screen.blit(surface, (10, y_offset))
            y_offset += 20

        # Draw waypoint details if waypoints exist
        if len(self.node.waypoints) > 0:
            y_offset += 10  # Add spacing
            title_surface = self.font.render("=== Waypoints ===", True, self.COLOR_TEXT)
            self.screen.blit(title_surface, (10, y_offset))
            y_offset += 20

            for i, waypoint in enumerate(self.node.waypoints):
                # Highlight current waypoint
                color = (255, 255, 0) if i == self.node.current_waypoint_index else self.COLOR_TEXT

                # Waypoint coordinates
                wp_text = f"WP{i + 1}: ({waypoint[0]:.2f}, {waypoint[1]:.2f})"
                wp_surface = self.font.render(wp_text, True, color)
                self.screen.blit(wp_surface, (10, y_offset))
                y_offset += 18

                # Distance to next waypoint
                next_idx = (i + 1) % len(self.node.waypoints)
                next_waypoint = self.node.waypoints[next_idx]
                dist = np.linalg.norm(next_waypoint - waypoint)
                dist_text = f"  → WP{next_idx + 1}: {dist:.2f}m"
                dist_surface = self.small_font.render(dist_text, True, (180, 180, 180))
                self.screen.blit(dist_surface, (10, y_offset))
                y_offset += 16

            # Total path length
            if len(self.node.waypoints) > 1:
                total_dist = 0.0
                for i in range(len(self.node.waypoints)):
                    next_idx = (i + 1) % len(self.node.waypoints)
                    total_dist += np.linalg.norm(self.node.waypoints[next_idx] - self.node.waypoints[i])

                y_offset += 5
                total_text = f"Total path: {total_dist:.2f}m"
                total_surface = self.font.render(total_text, True, (100, 255, 100))
                self.screen.blit(total_surface, (10, y_offset))
                y_offset += 20

        # Draw reset goal button
        button_color = self.COLOR_BUTTON_HOVER if self.reset_button_hovered else self.COLOR_BUTTON
        pygame.draw.rect(self.screen, button_color, self.reset_button_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, self.reset_button_rect, 2, border_radius=5)

        button_text = self.button_font.render("Reset Goal", True, self.COLOR_BUTTON_TEXT)
        text_rect = button_text.get_rect(center=self.reset_button_rect.center)
        self.screen.blit(button_text, text_rect)

        # Draw clear waypoints button
        clear_color = self.COLOR_BUTTON_HOVER if self.clear_button_hovered else self.COLOR_BUTTON
        pygame.draw.rect(self.screen, clear_color, self.clear_button_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, self.clear_button_rect, 2, border_radius=5)

        clear_text = self.button_font.render("Clear Waypoints", True, self.COLOR_BUTTON_TEXT)
        clear_text_rect = clear_text.get_rect(center=self.clear_button_rect.center)
        self.screen.blit(clear_text, clear_text_rect)

        # Draw Z-filter sliders
        self.draw_slider(self.z_min_slider_rect, self.node.z_min, self.z_slider_min, self.z_slider_max, "Z Min")
        self.draw_slider(self.z_max_slider_rect, self.node.z_max, self.z_slider_min, self.z_slider_max, "Z Max")

    def update_slider_value(self, rect: pygame.Rect, mouse_x: int) -> float:
        """Calculate slider value from mouse position"""
        relative_x = max(0, min(mouse_x - rect.x, rect.width))
        norm_value = relative_x / rect.width
        return self.z_slider_min + norm_value * (self.z_slider_max - self.z_slider_min)

    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if should quit."""
        mouse_pos = pygame.mouse.get_pos()
        self.reset_button_hovered = self.reset_button_rect.collidepoint(mouse_pos)
        self.clear_button_hovered = self.clear_button_rect.collidepoint(mouse_pos)

        # Check if shift is pressed
        keys = pygame.key.get_pressed()
        shift_pressed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    # Reset view
                    self.scale = 50.0
                    self.offset_x = self.width / 2
                    self.offset_y = self.height / 2
                    self.view_rotation = 0.0

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Check if clicking on reset button
                    if self.reset_button_rect.collidepoint(event.pos):
                        # Reset goal to robot's current position
                        self.node.set_goal(self.node.robot_pos[0], self.node.robot_pos[1])
                    # Check if clicking on clear waypoints button
                    elif self.clear_button_rect.collidepoint(event.pos):
                        self.node.clear_waypoints()
                    # Check if clicking on Z-min slider
                    elif self.z_min_slider_rect.collidepoint(event.pos):
                        self.dragging_z_min_slider = True
                        self.node.z_min = self.update_slider_value(self.z_min_slider_rect, event.pos[0])
                    # Check if clicking on Z-max slider
                    elif self.z_max_slider_rect.collidepoint(event.pos):
                        self.dragging_z_max_slider = True
                        self.node.z_max = self.update_slider_value(self.z_max_slider_rect, event.pos[0])
                    else:
                        # Set goal or add waypoint
                        world_x, world_y = self.screen_to_world(*event.pos)
                        if shift_pressed:
                            self.node.add_waypoint(world_x, world_y)
                        else:
                            self.node.set_goal(world_x, world_y)

                elif event.button == 2:  # Middle click - start pan
                    self.panning = True
                    self.last_mouse_pos = event.pos

                elif event.button == 3:  # Right click - start drag
                    self.dragging = True
                    self.last_mouse_pos = event.pos

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging_z_min_slider = False
                    self.dragging_z_max_slider = False
                elif event.button == 2:  # Middle release
                    self.panning = False
                    self.last_mouse_pos = None
                elif event.button == 3:  # Right release
                    self.dragging = False
                    self.last_mouse_pos = None

            elif event.type == pygame.MOUSEMOTION:
                if self.panning and self.last_mouse_pos:
                    # Pan view based on mouse movement
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    self.offset_x += dx
                    self.offset_y += dy
                    self.last_mouse_pos = event.pos
                elif self.dragging and self.last_mouse_pos:
                    # Rotate view based on horizontal drag
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    self.view_rotation += dx * 0.01  # Rotation sensitivity
                    self.last_mouse_pos = event.pos
                elif self.dragging_z_min_slider:
                    # Update Z-min slider
                    new_value = self.update_slider_value(self.z_min_slider_rect, event.pos[0])
                    self.node.z_min = min(new_value, self.node.z_max - 0.1)  # Keep min below max
                elif self.dragging_z_max_slider:
                    # Update Z-max slider
                    new_value = self.update_slider_value(self.z_max_slider_rect, event.pos[0])
                    self.node.z_max = max(new_value, self.node.z_min + 0.1)  # Keep max above min

            elif event.type == pygame.MOUSEWHEEL:
                # Zoom with mouse wheel
                zoom_factor = 1.1 if event.y > 0 else 0.9
                self.scale *= zoom_factor
                self.scale = max(10.0, min(200.0, self.scale))  # Clamp scale

        return True

    def run(self):
        """Main visualization loop"""
        running = True
        while running and rclpy.ok():
            running = self.handle_events()

            # Clear screen
            self.screen.fill(self.COLOR_BG)

            # Draw everything
            self.draw_grid()
            self.draw_point_cloud()
            self.draw_robot()
            self.draw_goal()
            self.draw_ui()

            # Update display
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS

            # Spin ROS
            rclpy.spin_once(self.node, timeout_sec=0.001)

        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Goal Control Visualization")
    args = parser.parse_args()

    rclpy.init()

    node = GoalControlNode()
    visualizer = GoalControlVisualizer(node)

    try:
        visualizer.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
