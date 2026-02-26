from __future__ import annotations

import heapq
import math
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker

Point2D = Tuple[float, float]


class PathFollowerNode(Node):
    def __init__(self) -> None:
        super().__init__("q_path_follower")

        self._declare_parameters()
        self._load_parameters()
        self._create_interfaces()
        self._init_state()

        self.create_timer(1.0 / self._control_rate_hz, self._on_control_timer)

        pose_source = "ground_truth" if self._use_ground_truth_pose else "odometry"
        self.get_logger().info(
            "Path follower ready. "
            f"path_topic={self._path_topic}, odom_topic={self._odom_topic}, "
            f"cmd_vel_topic={self._cmd_vel_topic}, map_topic={self._map_topic}, "
            f"pose_source={pose_source}, controller=heading_p"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("path_topic", "/q_learning/path")
        self.declare_parameter("odom_topic", "/model/vehicle_blue/odometry")
        self.declare_parameter("cmd_vel_topic", "/model/vehicle_blue/cmd_vel")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("robot_pose_topic", "/q_learning/robot_pose")
        self.declare_parameter("robot_body_topic", "/q_learning/robot_body")
        self.declare_parameter("frame_id", "map")

        self.declare_parameter("use_ground_truth_pose", False)
        self.declare_parameter(
            "ground_truth_pose_topic", "/world/q_learning_world/dynamic_pose/info"
        )
        self.declare_parameter("ground_truth_child_frame_id", "vehicle_blue")
        self.declare_parameter("odom_offset_x", 0.0)
        self.declare_parameter("odom_offset_y", 0.0)

        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("simplify_path", False)
        self.declare_parameter("waypoint_tolerance", 0.20)
        self.declare_parameter("goal_tolerance", 0.20)

        self.declare_parameter("min_linear_speed", 0.0)
        self.declare_parameter("max_linear_speed", 0.50)
        self.declare_parameter("max_angular_speed", 1.8)
        self.declare_parameter("linear_gain", 0.90)
        self.declare_parameter("angular_gain", 2.50)
        self.declare_parameter("turn_in_place_angle", 0.70)

        self.declare_parameter("robot_radius", 0.22)
        self.declare_parameter("obstacle_clearance", 0.10)
        self.declare_parameter("occupied_cell_half_size_ratio", 0.20)
        self.declare_parameter("safety_lookahead_time", 0.70)
        self.declare_parameter("safety_samples", 8)
        self.declare_parameter("escape_linear_speed", 0.10)
        self.declare_parameter("escape_reverse_speed", 0.06)
        self.declare_parameter("escape_heading_tolerance", 0.50)

    def _load_parameters(self) -> None:
        self._path_topic = str(self.get_parameter("path_topic").value)
        self._odom_topic = str(self.get_parameter("odom_topic").value)
        self._cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._map_topic = str(self.get_parameter("map_topic").value)
        self._robot_pose_topic = str(self.get_parameter("robot_pose_topic").value)
        self._robot_body_topic = str(self.get_parameter("robot_body_topic").value)
        self._frame_id = str(self.get_parameter("frame_id").value)

        self._use_ground_truth_pose = bool(
            self.get_parameter("use_ground_truth_pose").value
        )
        self._ground_truth_pose_topic = str(
            self.get_parameter("ground_truth_pose_topic").value
        )
        self._ground_truth_child_frame_id = str(
            self.get_parameter("ground_truth_child_frame_id").value
        )
        self._odom_offset_x = float(self.get_parameter("odom_offset_x").value)
        self._odom_offset_y = float(self.get_parameter("odom_offset_y").value)

        self._control_rate_hz = max(
            1e-3, float(self.get_parameter("control_rate_hz").value)
        )
        self._simplify_path_enabled = bool(self.get_parameter("simplify_path").value)
        self._waypoint_tolerance = max(
            0.02, float(self.get_parameter("waypoint_tolerance").value)
        )
        self._goal_tolerance = max(0.01, float(self.get_parameter("goal_tolerance").value))

        self._min_linear_speed = float(self.get_parameter("min_linear_speed").value)
        self._max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        if self._max_linear_speed < self._min_linear_speed:
            self._max_linear_speed = self._min_linear_speed
        self._max_angular_speed = abs(
            float(self.get_parameter("max_angular_speed").value)
        )
        self._linear_gain = max(0.0, float(self.get_parameter("linear_gain").value))
        self._angular_gain = max(0.0, float(self.get_parameter("angular_gain").value))
        self._turn_in_place_angle = self._clamp(
            float(self.get_parameter("turn_in_place_angle").value), 0.0, math.pi
        )

        self._robot_radius = max(0.05, float(self.get_parameter("robot_radius").value))
        self._obstacle_clearance = max(
            0.0, float(self.get_parameter("obstacle_clearance").value)
        )
        self._occupied_cell_half_size_ratio = self._clamp(
            float(self.get_parameter("occupied_cell_half_size_ratio").value), 0.0, 0.5
        )
        self._safety_lookahead_time = max(
            0.0, float(self.get_parameter("safety_lookahead_time").value)
        )
        self._safety_samples = max(1, int(self.get_parameter("safety_samples").value))
        self._escape_linear_speed = max(
            0.0, float(self.get_parameter("escape_linear_speed").value)
        )
        self._escape_reverse_speed = max(
            0.0, float(self.get_parameter("escape_reverse_speed").value)
        )
        self._escape_heading_tolerance = self._clamp(
            float(self.get_parameter("escape_heading_tolerance").value), 0.05, math.pi
        )

    def _create_interfaces(self) -> None:
        self._cmd_pub = self.create_publisher(Twist, self._cmd_vel_topic, 10)
        self._robot_pose_pub = self.create_publisher(
            PoseStamped, self._robot_pose_topic, 10
        )
        self._robot_body_pub = self.create_publisher(Marker, self._robot_body_topic, 10)

        self.create_subscription(Path, self._path_topic, self._on_path, 10)
        self.create_subscription(Odometry, self._odom_topic, self._on_odom, 10)
        if self._use_ground_truth_pose:
            self.create_subscription(
                TFMessage,
                self._ground_truth_pose_topic,
                self._on_ground_truth_pose,
                10,
            )
        self.create_subscription(OccupancyGrid, self._map_topic, self._on_map, 10)

    def _init_state(self) -> None:
        self._current_path: List[Point2D] = []
        self._active_waypoint_index = 0
        self._robot_pose: Optional[Tuple[float, float, float]] = None
        self._goal_reached_logged = False

        self._map_ready = False
        self._map_resolution = 1.0
        self._map_origin_x = 0.0
        self._map_origin_y = 0.0
        self._map_width = 0
        self._map_height = 0
        self._map_distance: List[List[float]] = []

        self._last_no_map_warn_ns = 0
        self._last_blocked_warn_ns = 0
        self._last_collision_warn_ns = 0
        self._last_ground_truth_missing_log_ns = 0

    def _on_path(self, msg: Path) -> None:
        raw_path = [
            (float(p.pose.position.x), float(p.pose.position.y)) for p in msg.poses
        ]
        if self._simplify_path_enabled:
            self._current_path = self._simplify_path(raw_path)
            path_mode = "simplified"
        else:
            self._current_path = raw_path
            path_mode = "raw"
        self._active_waypoint_index = 0
        self._goal_reached_logged = False

        if not self._current_path:
            self._publish_stop()
            self.get_logger().warning("Received empty path; robot stopped.")
            return

        self.get_logger().info(
            f"Received path with {len(raw_path)} waypoints; "
            f"{path_mode} waypoint count={len(self._current_path)}."
        )

    def _on_odom(self, msg: Odometry) -> None:
        if self._use_ground_truth_pose:
            return

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = self._yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self._update_robot_pose(
            x=float(p.x) + self._odom_offset_x,
            y=float(p.y) + self._odom_offset_y,
            qx=float(q.x),
            qy=float(q.y),
            qz=float(q.z),
            qw=float(q.w),
            yaw=yaw,
            stamp=msg.header.stamp,
        )

    def _on_ground_truth_pose(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            if transform.child_frame_id != self._ground_truth_child_frame_id:
                continue

            t = transform.transform.translation
            q = transform.transform.rotation
            yaw = self._yaw_from_quaternion(q.x, q.y, q.z, q.w)
            self._update_robot_pose(
                x=float(t.x),
                y=float(t.y),
                qx=float(q.x),
                qy=float(q.y),
                qz=float(q.z),
                qw=float(q.w),
                yaw=yaw,
                stamp=transform.header.stamp,
            )
            return

        now_ns = self.get_clock().now().nanoseconds
        if (now_ns - self._last_ground_truth_missing_log_ns) > 2_000_000_000:
            self._last_ground_truth_missing_log_ns = now_ns
            self.get_logger().warning(
                "Ground-truth message missing child_frame_id "
                f"'{self._ground_truth_child_frame_id}'."
            )

    def _update_robot_pose(
        self,
        x: float,
        y: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
        yaw: float,
        stamp,
    ) -> None:
        self._robot_pose = (x, y, yaw)

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self._frame_id
        if int(stamp.sec) == 0 and int(stamp.nanosec) == 0:
            pose_msg.header.stamp = self.get_clock().now().to_msg()
        else:
            pose_msg.header.stamp = stamp
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        self._robot_pose_pub.publish(pose_msg)

        marker = Marker()
        marker.header = pose_msg.header
        marker.ns = "q_learning_robot"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose = pose_msg.pose
        marker.pose.position.z = 0.08
        marker.scale.x = 0.35
        marker.scale.y = 0.24
        marker.scale.z = 0.12
        marker.color.r = 0.2
        marker.color.g = 0.55
        marker.color.b = 1.0
        marker.color.a = 0.95
        self._robot_body_pub.publish(marker)

    def _on_map(self, msg: OccupancyGrid) -> None:
        width = int(msg.info.width)
        height = int(msg.info.height)
        if width <= 0 or height <= 0:
            return

        self._map_width = width
        self._map_height = height
        self._map_resolution = float(msg.info.resolution)
        self._map_origin_x = float(msg.info.origin.position.x)
        self._map_origin_y = float(msg.info.origin.position.y)

        occupancy: List[List[bool]] = [[False for _ in range(width)] for _ in range(height)]
        for y in range(height):
            row_offset = y * width
            for x in range(width):
                value = int(msg.data[row_offset + x])
                occupancy[y][x] = value < 0 or value >= 50

        self._map_distance = self._build_obstacle_distance_map(occupancy)
        self._map_ready = True

    def _on_control_timer(self) -> None:
        if self._robot_pose is None or not self._current_path:
            return

        if not self._map_ready:
            self._publish_stop()
            self._throttled_warning(
                "Waiting for /map before running path follower.", "no_map"
            )
            return

        x, y, yaw = self._robot_pose
        goal_x, goal_y = self._current_path[-1]
        distance_to_goal = math.hypot(goal_x - x, goal_y - y)

        if distance_to_goal <= self._goal_tolerance:
            self._publish_stop()
            self._current_path = []
            if not self._goal_reached_logged:
                self.get_logger().info("Goal reached; robot stopped.")
                self._goal_reached_logged = True
            return

        self._update_waypoint_index(x, y)
        if self._active_waypoint_index >= len(self._current_path):
            self._publish_stop()
            return
        target_x, target_y = self._current_path[self._active_waypoint_index]
        cmd_pair = self._compute_tracking_command(
            pose=(x, y, yaw),
            target=(target_x, target_y),
        )

        cmd = Twist()
        cmd.linear.x = cmd_pair[0]
        cmd.angular.z = cmd_pair[1]
        self._cmd_pub.publish(cmd)

    def _update_waypoint_index(self, x: float, y: float) -> None:
        if not self._current_path:
            return

        if self._active_waypoint_index >= len(self._current_path):
            return

        # Keep strict sequential waypoint following:
        # do not jump to nearest future waypoint before current one is reached.
        while self._active_waypoint_index < len(self._current_path) - 1:
            wx, wy = self._current_path[self._active_waypoint_index]
            if math.hypot(wx - x, wy - y) > self._waypoint_tolerance:
                break
            self._active_waypoint_index += 1

    def _compute_tracking_command(
        self,
        pose: Tuple[float, float, float],
        target: Point2D,
    ) -> Tuple[float, float]:
        x, y, yaw = pose
        target_dx = target[0] - x
        target_dy = target[1] - y
        target_distance = math.hypot(target_dx, target_dy)
        if target_distance <= 1e-6:
            return (0.0, 0.0)

        desired_heading = math.atan2(target_dy, target_dx)
        heading_error = self._normalize_angle(desired_heading - yaw)

        angular_cmd = self._clamp(
            self._angular_gain * heading_error,
            -self._max_angular_speed,
            self._max_angular_speed,
        )

        alignment = max(0.0, math.cos(heading_error))
        linear_cmd = self._linear_gain * target_distance * alignment
        linear_cmd = self._clamp(linear_cmd, 0.0, self._max_linear_speed)

        if abs(heading_error) > self._turn_in_place_angle:
            linear_cmd = 0.0
        elif linear_cmd > 0.0 and self._min_linear_speed > 0.0:
            linear_cmd = max(linear_cmd, self._min_linear_speed)

        collision_now, clearance_now = self._collision_and_clearance(x, y)
        if collision_now:
            self._throttled_warning(
                f"Robot is inside obstacle safety margin (clearance={clearance_now:.3f} m); "
                "running escape behavior.",
                "in_collision",
            )
            return self._compute_escape_command(x, y, yaw)

        if linear_cmd > 1e-4 and self._safety_lookahead_time > 1e-6:
            linear_cmd = self._find_safe_linear_speed(
                x=x,
                y=y,
                yaw=yaw,
                angular=angular_cmd,
                initial_linear=linear_cmd,
            )
            if linear_cmd <= 1e-5:
                self._throttled_warning(
                    "Forward motion blocked by obstacle ahead; stopping translation.",
                    "blocked",
                )

        return (linear_cmd, angular_cmd)

    def _compute_escape_command(
        self, x: float, y: float, yaw: float
    ) -> Tuple[float, float]:
        cell = self._world_to_cell(x, y)
        if cell is None:
            return (0.0, 0.0)

        cx, cy = cell
        current_clearance = self._map_distance[cy][cx]
        best_clearance = current_clearance
        best_neighbor: Optional[Tuple[int, int]] = None

        for dx, dy in (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ):
            nx = cx + dx
            ny = cy + dy
            if nx < 0 or ny < 0 or nx >= self._map_width or ny >= self._map_height:
                continue
            clearance = self._map_distance[ny][nx]
            if clearance > best_clearance + 1e-6:
                best_clearance = clearance
                best_neighbor = (nx, ny)

        if best_neighbor is None:
            return (0.0, 0.0)

        target_x, target_y = self._cell_center(*best_neighbor)
        escape_heading = math.atan2(target_y - y, target_x - x)
        heading_error = self._normalize_angle(escape_heading - yaw)
        angular_cmd = self._clamp(
            self._angular_gain * heading_error,
            -self._max_angular_speed,
            self._max_angular_speed,
        )

        linear_cmd = 0.0
        if abs(heading_error) <= self._escape_heading_tolerance:
            linear_cmd = self._escape_linear_speed
        elif abs(heading_error) >= (math.pi - self._escape_heading_tolerance):
            linear_cmd = -self._escape_reverse_speed

        if (
            abs(linear_cmd) > 1e-5
            and self._safety_lookahead_time > 1e-6
            and self._would_collide(x, y, yaw, linear_cmd, angular_cmd)
        ):
            linear_cmd = 0.0

        return (linear_cmd, angular_cmd)

    def _find_safe_linear_speed(
        self,
        x: float,
        y: float,
        yaw: float,
        angular: float,
        initial_linear: float,
    ) -> float:
        linear = initial_linear
        for _ in range(8):
            if not self._would_collide(x, y, yaw, linear, angular):
                return linear
            if linear <= 0.02:
                return 0.0
            linear *= 0.5
        return 0.0

    def _would_collide(
        self, x: float, y: float, yaw: float, linear: float, angular: float
    ) -> bool:
        steps = max(1, self._safety_samples)
        dt = self._safety_lookahead_time / float(steps)
        px = x
        py = y
        pyaw = yaw

        for _ in range(steps):
            px += linear * math.cos(pyaw) * dt
            py += linear * math.sin(pyaw) * dt
            pyaw = self._normalize_angle(pyaw + angular * dt)
            collision, _ = self._collision_and_clearance(px, py)
            if collision:
                return True
        return False

    def _simplify_path(self, points: List[Point2D]) -> List[Point2D]:
        if len(points) <= 2:
            return points

        simplified: List[Point2D] = [points[0]]
        for i in range(1, len(points) - 1):
            prev = simplified[-1]
            cur = points[i]
            nxt = points[i + 1]
            dir1 = self._segment_direction(prev, cur)
            dir2 = self._segment_direction(cur, nxt)
            if dir1 == (0, 0) or dir2 == (0, 0):
                continue
            if dir1 == dir2:
                continue
            simplified.append(cur)
        simplified.append(points[-1])
        return simplified

    @staticmethod
    def _segment_direction(first: Point2D, second: Point2D) -> Tuple[int, int]:
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        eps = 1e-6
        sx = 0 if abs(dx) < eps else (1 if dx > 0.0 else -1)
        sy = 0 if abs(dy) < eps else (1 if dy > 0.0 else -1)
        return (sx, sy)

    def _collision_and_clearance(self, x: float, y: float) -> Tuple[bool, float]:
        cell = self._world_to_cell(x, y)
        if cell is None:
            return True, 0.0

        cx, cy = cell
        clearance_cells = self._map_distance[cy][cx]
        center_distance_m = clearance_cells * self._map_resolution
        cell_half_size_m = self._occupied_cell_half_size_ratio * self._map_resolution
        clearance_m = max(0.0, center_distance_m - cell_half_size_m)
        collision = clearance_m < (self._robot_radius + self._obstacle_clearance)
        return collision, clearance_m

    def _world_to_cell(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if not self._map_ready:
            return None
        cx = int(math.floor((x - self._map_origin_x) / self._map_resolution))
        cy = int(math.floor((y - self._map_origin_y) / self._map_resolution))
        if cx < 0 or cy < 0 or cx >= self._map_width or cy >= self._map_height:
            return None
        return (cx, cy)

    def _cell_center(self, cx: int, cy: int) -> Tuple[float, float]:
        return (
            self._map_origin_x + (float(cx) + 0.5) * self._map_resolution,
            self._map_origin_y + (float(cy) + 0.5) * self._map_resolution,
        )

    def _build_obstacle_distance_map(
        self, occupancy: List[List[bool]]
    ) -> List[List[float]]:
        height = len(occupancy)
        width = len(occupancy[0]) if height else 0
        inf_dist = float(width + height + 1)
        distance = [[inf_dist for _ in range(width)] for _ in range(height)]
        queue: List[Tuple[float, int, int]] = []

        for y in range(height):
            for x in range(width):
                if occupancy[y][x]:
                    distance[y][x] = 0.0
                    heapq.heappush(queue, (0.0, x, y))

        neighbors = (
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (-1, -1, math.sqrt(2.0)),
        )
        while queue:
            current_dist, x, y = heapq.heappop(queue)
            if current_dist > distance[y][x] + 1e-9:
                continue
            for dx, dy, step_cost in neighbors:
                nx = x + dx
                ny = y + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                next_dist = current_dist + step_cost
                if next_dist >= distance[ny][nx]:
                    continue
                distance[ny][nx] = next_dist
                heapq.heappush(queue, (next_dist, nx, ny))

        return distance

    def _throttled_warning(self, message: str, warn_key: str) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if warn_key == "no_map":
            if now_ns - self._last_no_map_warn_ns < 2_000_000_000:
                return
            self._last_no_map_warn_ns = now_ns
        elif warn_key == "blocked":
            if now_ns - self._last_blocked_warn_ns < 2_000_000_000:
                return
            self._last_blocked_warn_ns = now_ns
        elif warn_key == "in_collision":
            if now_ns - self._last_collision_warn_ns < 2_000_000_000:
                return
            self._last_collision_warn_ns = now_ns
        self.get_logger().warning(message)

    def _publish_stop(self) -> None:
        self._cmd_pub.publish(Twist())

    @staticmethod
    def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, value))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PathFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
