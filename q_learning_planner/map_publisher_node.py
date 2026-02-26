from __future__ import annotations

import os
from typing import List

import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from .grid_map import GridMap


class MapPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("q_map_publisher")

        default_map = os.path.join(
            get_package_share_directory("q_learning_planner"),
            "config",
            "generated_grid_map.json",
        )

        self.declare_parameter("map_config_path", default_map)
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_period_sec", 1.0)

        map_path = str(self.get_parameter("map_config_path").value)
        map_topic = str(self.get_parameter("map_topic").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        publish_period = float(self.get_parameter("publish_period_sec").value)

        grid_map = GridMap.from_json(map_path)
        self._map_msg = self._build_occupancy_grid(grid_map)

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._publisher = self.create_publisher(OccupancyGrid, map_topic, qos)
        self.create_timer(max(0.1, publish_period), self._publish_map)

        self.get_logger().info(
            f"Publishing occupancy grid from '{map_path}' on '{map_topic}'. "
            f"Size={grid_map.width}x{grid_map.height}, resolution={grid_map.resolution}"
        )

    def _publish_map(self) -> None:
        now_msg = self.get_clock().now().to_msg()
        self._map_msg.header.stamp = now_msg
        self._map_msg.info.map_load_time = now_msg
        self._publisher.publish(self._map_msg)

    def _build_occupancy_grid(self, grid_map: GridMap) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.frame_id = self._frame_id
        msg.info.resolution = float(grid_map.resolution)
        msg.info.width = int(grid_map.width)
        msg.info.height = int(grid_map.height)
        msg.info.origin.position.x = float(grid_map.origin[0])
        msg.info.origin.position.y = float(grid_map.origin[1])
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        data: List[int] = []
        for y in range(grid_map.height):
            for x in range(grid_map.width):
                data.append(100 if grid_map.occupancy[y, x] else 0)
        msg.data = data
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MapPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
