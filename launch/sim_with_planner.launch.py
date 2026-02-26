import json
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("q_learning_planner")
    default_world = os.path.join(package_share, "worlds", "q_learning_world.sdf")
    default_map_path = os.path.join(package_share, "config", "generated_grid_map.json")
    gazebo_launch_file = os.path.join(package_share, "launch", "gazebo_world.launch.py")
    default_rviz_config = os.path.join(package_share, "rviz", "q_learning_planner.rviz")

    layout_file = os.path.join(package_share, "config", "environment_layout.json")
    default_odom_offset_x = 0.0
    default_odom_offset_y = 0.0
    if os.path.exists(layout_file):
        with open(layout_file, "r", encoding="utf-8") as file:
            layout = json.load(file)
        robot_start = layout.get("robot_start")
        if isinstance(robot_start, dict) and "x" in robot_start and "y" in robot_start:
            default_odom_offset_x = float(robot_start["x"])
            default_odom_offset_y = float(robot_start["y"])
        else:
            resolution = float(layout.get("resolution", 1.0))
            origin = layout.get("origin", [0.0, 0.0])
            start_cell = layout.get("robot_start_cell", [0, 0])
            default_odom_offset_x = float(origin[0]) + (
                float(start_cell[0]) + 0.5
            ) * resolution
            default_odom_offset_y = float(origin[1]) + (
                float(start_cell[1]) + 0.5
            ) * resolution

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file),
        launch_arguments={"world": LaunchConfiguration("world")}.items(),
    )

    planner_node = Node(
        package="q_learning_planner",
        executable="q_path_planner",
        output="screen",
        parameters=[
            {
                "model_path": LaunchConfiguration("model_path"),
                "path_topic": LaunchConfiguration("path_topic"),
                "goal_topic": LaunchConfiguration("goal_topic"),
                "use_ground_truth_start": ParameterValue(
                    LaunchConfiguration("use_ground_truth_start"), value_type=bool
                ),
                "ground_truth_pose_topic": LaunchConfiguration(
                    "ground_truth_pose_topic"
                ),
                "ground_truth_child_frame_id": LaunchConfiguration(
                    "ground_truth_child_frame_id"
                ),
                "use_odom_start": ParameterValue(
                    LaunchConfiguration("use_odom_start"), value_type=bool
                ),
                "use_initialpose_start": False,
                "odom_topic": LaunchConfiguration("odom_topic"),
                "odom_offset_x": ParameterValue(
                    LaunchConfiguration("odom_offset_x"), value_type=float
                ),
                "odom_offset_y": ParameterValue(
                    LaunchConfiguration("odom_offset_y"), value_type=float
                ),
                "snap_start_to_nearest_free_cell": ParameterValue(
                    LaunchConfiguration("snap_start_to_nearest_free_cell"),
                    value_type=bool,
                ),
                "snap_to_nearest_free_cell": ParameterValue(
                    LaunchConfiguration("snap_to_nearest_free_cell"), value_type=bool
                ),
                "max_planning_steps": ParameterValue(
                    LaunchConfiguration("max_planning_steps"), value_type=int
                ),
            }
        ],
    )

    follower_node = Node(
        package="q_learning_planner",
        executable="q_path_follower",
        output="screen",
        parameters=[
            {
                "path_topic": LaunchConfiguration("path_topic"),
                "odom_topic": LaunchConfiguration("odom_topic"),
                "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
                "use_ground_truth_pose": ParameterValue(
                    LaunchConfiguration("use_ground_truth_pose"), value_type=bool
                ),
                "ground_truth_pose_topic": LaunchConfiguration(
                    "ground_truth_pose_topic"
                ),
                "ground_truth_child_frame_id": LaunchConfiguration(
                    "ground_truth_child_frame_id"
                ),
                "map_topic": LaunchConfiguration("map_topic"),
                "robot_pose_topic": LaunchConfiguration("robot_pose_topic"),
                "robot_body_topic": LaunchConfiguration("robot_body_topic"),
                "frame_id": "map",
                "odom_offset_x": ParameterValue(
                    LaunchConfiguration("odom_offset_x"), value_type=float
                ),
                "odom_offset_y": ParameterValue(
                    LaunchConfiguration("odom_offset_y"), value_type=float
                ),
                "control_rate_hz": ParameterValue(
                    LaunchConfiguration("control_rate_hz"), value_type=float
                ),
                "simplify_path": ParameterValue(
                    LaunchConfiguration("simplify_path"), value_type=bool
                ),
                "waypoint_tolerance": ParameterValue(
                    LaunchConfiguration("waypoint_tolerance"), value_type=float
                ),
                "goal_tolerance": ParameterValue(
                    LaunchConfiguration("goal_tolerance"), value_type=float
                ),
                "min_linear_speed": ParameterValue(
                    LaunchConfiguration("min_linear_speed"), value_type=float
                ),
                "max_linear_speed": ParameterValue(
                    LaunchConfiguration("max_linear_speed"), value_type=float
                ),
                "max_angular_speed": ParameterValue(
                    LaunchConfiguration("max_angular_speed"), value_type=float
                ),
                "linear_gain": ParameterValue(
                    LaunchConfiguration("linear_gain"), value_type=float
                ),
                "angular_gain": ParameterValue(
                    LaunchConfiguration("angular_gain"), value_type=float
                ),
                "turn_in_place_angle": ParameterValue(
                    LaunchConfiguration("turn_in_place_angle"), value_type=float
                ),
                "robot_radius": ParameterValue(
                    LaunchConfiguration("robot_radius"), value_type=float
                ),
                "obstacle_clearance": ParameterValue(
                    LaunchConfiguration("obstacle_clearance"), value_type=float
                ),
                "occupied_cell_half_size_ratio": ParameterValue(
                    LaunchConfiguration("occupied_cell_half_size_ratio"),
                    value_type=float,
                ),
                "safety_lookahead_time": ParameterValue(
                    LaunchConfiguration("safety_lookahead_time"), value_type=float
                ),
                "safety_samples": ParameterValue(
                    LaunchConfiguration("safety_samples"), value_type=int
                ),
                "escape_linear_speed": ParameterValue(
                    LaunchConfiguration("escape_linear_speed"), value_type=float
                ),
                "escape_reverse_speed": ParameterValue(
                    LaunchConfiguration("escape_reverse_speed"), value_type=float
                ),
                "escape_heading_tolerance": ParameterValue(
                    LaunchConfiguration("escape_heading_tolerance"), value_type=float
                ),
            }
        ],
    )

    map_publisher_node = Node(
        package="q_learning_planner",
        executable="q_map_publisher",
        output="screen",
        parameters=[
            {
                "map_config_path": LaunchConfiguration("map_config_path"),
                "map_topic": "/map",
                "frame_id": "map",
                "publish_period_sec": 1.0,
            }
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        output="screen",
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "world",
                default_value=default_world,
                description="Gazebo world file path.",
            ),
            DeclareLaunchArgument(
                "model_path",
                default_value="/tmp/q_learning_q_tables.npz",
                description="Trained Q-table model path.",
            ),
            DeclareLaunchArgument(
                "map_config_path",
                default_value=default_map_path,
                description="Occupancy map JSON used for RViz map display.",
            ),
            DeclareLaunchArgument("path_topic", default_value="/q_learning/path"),
            DeclareLaunchArgument("goal_topic", default_value="/goal_pose"),
            DeclareLaunchArgument("map_topic", default_value="/map"),
            DeclareLaunchArgument("use_ground_truth_start", default_value="true"),
            DeclareLaunchArgument("use_odom_start", default_value="false"),
            DeclareLaunchArgument("use_ground_truth_pose", default_value="true"),
            DeclareLaunchArgument(
                "ground_truth_pose_topic",
                default_value="/world/q_learning_world/dynamic_pose/info",
            ),
            DeclareLaunchArgument(
                "ground_truth_child_frame_id",
                default_value="vehicle_blue",
            ),
            DeclareLaunchArgument(
                "odom_topic", default_value="/model/vehicle_blue/odometry"
            ),
            DeclareLaunchArgument(
                "odom_offset_x", default_value=str(default_odom_offset_x)
            ),
            DeclareLaunchArgument(
                "odom_offset_y", default_value=str(default_odom_offset_y)
            ),
            DeclareLaunchArgument(
                "cmd_vel_topic", default_value="/model/vehicle_blue/cmd_vel"
            ),
            DeclareLaunchArgument(
                "robot_pose_topic", default_value="/q_learning/robot_pose"
            ),
            DeclareLaunchArgument(
                "robot_body_topic", default_value="/q_learning/robot_body"
            ),
            DeclareLaunchArgument(
                "snap_start_to_nearest_free_cell", default_value="true"
            ),
            DeclareLaunchArgument("snap_to_nearest_free_cell", default_value="true"),
            DeclareLaunchArgument("max_planning_steps", default_value="500"),
            DeclareLaunchArgument("control_rate_hz", default_value="20.0"),
            DeclareLaunchArgument("simplify_path", default_value="false"),
            DeclareLaunchArgument("waypoint_tolerance", default_value="0.20"),
            DeclareLaunchArgument("goal_tolerance", default_value="0.20"),
            DeclareLaunchArgument("min_linear_speed", default_value="0.0"),
            DeclareLaunchArgument("max_linear_speed", default_value="0.50"),
            DeclareLaunchArgument("max_angular_speed", default_value="1.8"),
            DeclareLaunchArgument("linear_gain", default_value="0.90"),
            DeclareLaunchArgument("angular_gain", default_value="2.50"),
            DeclareLaunchArgument("turn_in_place_angle", default_value="0.70"),
            DeclareLaunchArgument("robot_radius", default_value="0.22"),
            DeclareLaunchArgument("obstacle_clearance", default_value="0.10"),
            DeclareLaunchArgument(
                "occupied_cell_half_size_ratio", default_value="0.20"
            ),
            DeclareLaunchArgument("safety_lookahead_time", default_value="0.70"),
            DeclareLaunchArgument("safety_samples", default_value="8"),
            DeclareLaunchArgument("escape_linear_speed", default_value="0.10"),
            DeclareLaunchArgument("escape_reverse_speed", default_value="0.06"),
            DeclareLaunchArgument("escape_heading_tolerance", default_value="0.50"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("rviz_config", default_value=default_rviz_config),
            gazebo_launch,
            map_publisher_node,
            planner_node,
            follower_node,
            rviz_node,
        ]
    )
