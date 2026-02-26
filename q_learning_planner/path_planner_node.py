from __future__ import annotations

from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node

from .grid_map import Cell
from .q_learning_core import QTablePlanner, load_q_table_model


class QPathPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("q_path_planner")

        self.declare_parameter("model_path", "/tmp/q_learning_q_tables.npz")
        self.declare_parameter("path_topic", "/q_learning/path")
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("initial_pose_topic", "/initialpose")
        self.declare_parameter("use_initialpose_start", True)
        self.declare_parameter("use_odom_start", False)
        self.declare_parameter("odom_topic", "/model/vehicle_blue/odometry")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("max_planning_steps", 500)
        self.declare_parameter("allow_bfs_fallback", True)
        self.declare_parameter("plan_on_startup", False)
        self.declare_parameter("start_cell_x", 0)
        self.declare_parameter("start_cell_y", 0)
        self.declare_parameter("goal_cell_x", 5)
        self.declare_parameter("goal_cell_y", 5)

        model_path = str(self.get_parameter("model_path").value)
        self._planner = QTablePlanner(load_q_table_model(model_path))

        self._frame_id = str(self.get_parameter("frame_id").value)
        self._max_planning_steps = int(self.get_parameter("max_planning_steps").value)
        self._allow_bfs_fallback = bool(self.get_parameter("allow_bfs_fallback").value)
        self._use_odom_start = bool(self.get_parameter("use_odom_start").value)
        self._use_initialpose_start = bool(
            self.get_parameter("use_initialpose_start").value
        )

        path_topic = str(self.get_parameter("path_topic").value)
        goal_topic = str(self.get_parameter("goal_topic").value)
        initial_pose_topic = str(self.get_parameter("initial_pose_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)

        self._path_pub = self.create_publisher(Path, path_topic, 10)
        self.create_subscription(PoseStamped, goal_topic, self._on_goal_pose, 10)

        self._latest_initial_cell: Optional[Cell] = None
        self._latest_odom_cell: Optional[Cell] = None

        if self._use_initialpose_start:
            self.create_subscription(
                PoseWithCovarianceStamped,
                initial_pose_topic,
                self._on_initial_pose,
                10,
            )
            self.get_logger().info(
                f"Using start pose from '{initial_pose_topic}' when available."
            )

        if self._use_odom_start:
            self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
            self.get_logger().info(
                f"Using start pose from odometry topic '{odom_topic}'."
            )

        plan_on_startup = bool(self.get_parameter("plan_on_startup").value)
        self._startup_timer = None
        if plan_on_startup:
            self._startup_timer = self.create_timer(0.5, self._on_startup_timer)

        self.get_logger().info(
            f"Loaded Q-table model '{model_path}'. "
            f"Path publisher topic: '{path_topic}'."
        )

    def _on_startup_timer(self) -> None:
        if self._startup_timer is not None:
            self._startup_timer.cancel()
            self._startup_timer = None

        start_cell = (
            int(self.get_parameter("start_cell_x").value),
            int(self.get_parameter("start_cell_y").value),
        )
        goal_cell = (
            int(self.get_parameter("goal_cell_x").value),
            int(self.get_parameter("goal_cell_y").value),
        )
        self._plan_and_publish(start_cell, goal_cell, trigger="startup")

    def _on_initial_pose(self, msg: PoseWithCovarianceStamped) -> None:
        pose = msg.pose.pose.position
        self._latest_initial_cell = self._planner.model.grid_map.world_to_cell(
            pose.x, pose.y
        )
        self.get_logger().info(
            f"Updated start cell from /initialpose: {self._latest_initial_cell}"
        )

    def _on_odom(self, msg: Odometry) -> None:
        pose = msg.pose.pose.position
        self._latest_odom_cell = self._planner.model.grid_map.world_to_cell(
            pose.x, pose.y
        )

    def _on_goal_pose(self, msg: PoseStamped) -> None:
        goal_position = msg.pose.position
        goal_cell = self._planner.model.grid_map.world_to_cell(
            goal_position.x, goal_position.y
        )
        start_cell = self._resolve_start_cell()

        if start_cell is None:
            self.get_logger().warning("No start cell available yet; ignoring goal.")
            return

        self._plan_and_publish(start_cell, goal_cell, trigger="goal_topic")

    def _resolve_start_cell(self) -> Optional[Cell]:
        if self._use_odom_start and self._latest_odom_cell is not None:
            return self._latest_odom_cell
        if self._use_initialpose_start and self._latest_initial_cell is not None:
            return self._latest_initial_cell

        return (
            int(self.get_parameter("start_cell_x").value),
            int(self.get_parameter("start_cell_y").value),
        )

    def _plan_and_publish(self, start_cell: Cell, goal_cell: Cell, trigger: str) -> None:
        result = self._planner.plan(
            start_cell=start_cell,
            goal_cell=goal_cell,
            max_steps=self._max_planning_steps,
            allow_fallback=self._allow_bfs_fallback,
        )

        if not result.path:
            self.get_logger().error(
                f"Planning failed ({trigger}). start={start_cell}, goal={goal_cell}"
            )
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self._frame_id

        for cell in result.path:
            pose = PoseStamped()
            pose.header = path_msg.header
            world_x, world_y = self._planner.model.grid_map.cell_center(cell)
            pose.pose.position.x = world_x
            pose.pose.position.y = world_y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self._path_pub.publish(path_msg)

        method = "RL" if not result.used_fallback else "BFS fallback"
        self.get_logger().info(
            f"Published path with {len(result.path)} waypoints ({method}). "
            f"start={start_cell}, goal={goal_cell}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = QPathPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
