from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model_path",
                default_value="/tmp/q_learning_q_tables.npz",
                description="Path to a trained Q-table model (.npz).",
            ),
            DeclareLaunchArgument("path_topic", default_value="/q_learning/path"),
            DeclareLaunchArgument("goal_topic", default_value="/goal_pose"),
            DeclareLaunchArgument("initial_pose_topic", default_value="/initialpose"),
            DeclareLaunchArgument("use_initialpose_start", default_value="true"),
            DeclareLaunchArgument("use_ground_truth_start", default_value="false"),
            DeclareLaunchArgument(
                "ground_truth_pose_topic",
                default_value="/world/q_learning_world/dynamic_pose/info",
            ),
            DeclareLaunchArgument(
                "ground_truth_child_frame_id",
                default_value="vehicle_blue",
            ),
            DeclareLaunchArgument("use_odom_start", default_value="false"),
            DeclareLaunchArgument(
                "odom_topic", default_value="/model/vehicle_blue/odometry"
            ),
            DeclareLaunchArgument("odom_offset_x", default_value="0.0"),
            DeclareLaunchArgument("odom_offset_y", default_value="0.0"),
            DeclareLaunchArgument("frame_id", default_value="map"),
            DeclareLaunchArgument("max_planning_steps", default_value="500"),
            DeclareLaunchArgument(
                "snap_start_to_nearest_free_cell", default_value="true"
            ),
            DeclareLaunchArgument("snap_to_nearest_free_cell", default_value="true"),
            DeclareLaunchArgument("plan_on_startup", default_value="false"),
            DeclareLaunchArgument("start_cell_x", default_value="0"),
            DeclareLaunchArgument("start_cell_y", default_value="0"),
            DeclareLaunchArgument("goal_cell_x", default_value="10"),
            DeclareLaunchArgument("goal_cell_y", default_value="10"),
            Node(
                package="q_learning_planner",
                executable="q_path_planner",
                output="screen",
                parameters=[
                    {
                        "model_path": LaunchConfiguration("model_path"),
                        "path_topic": LaunchConfiguration("path_topic"),
                        "goal_topic": LaunchConfiguration("goal_topic"),
                        "initial_pose_topic": LaunchConfiguration(
                            "initial_pose_topic"
                        ),
                        "use_initialpose_start": ParameterValue(
                            LaunchConfiguration("use_initialpose_start"),
                            value_type=bool,
                        ),
                        "use_ground_truth_start": ParameterValue(
                            LaunchConfiguration("use_ground_truth_start"),
                            value_type=bool,
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
                        "odom_topic": LaunchConfiguration("odom_topic"),
                        "odom_offset_x": ParameterValue(
                            LaunchConfiguration("odom_offset_x"), value_type=float
                        ),
                        "odom_offset_y": ParameterValue(
                            LaunchConfiguration("odom_offset_y"), value_type=float
                        ),
                        "frame_id": LaunchConfiguration("frame_id"),
                        "max_planning_steps": ParameterValue(
                            LaunchConfiguration("max_planning_steps"), value_type=int
                        ),
                        "snap_start_to_nearest_free_cell": ParameterValue(
                            LaunchConfiguration("snap_start_to_nearest_free_cell"),
                            value_type=bool,
                        ),
                        "snap_to_nearest_free_cell": ParameterValue(
                            LaunchConfiguration("snap_to_nearest_free_cell"),
                            value_type=bool,
                        ),
                        "plan_on_startup": ParameterValue(
                            LaunchConfiguration("plan_on_startup"), value_type=bool
                        ),
                        "start_cell_x": ParameterValue(
                            LaunchConfiguration("start_cell_x"), value_type=int
                        ),
                        "start_cell_y": ParameterValue(
                            LaunchConfiguration("start_cell_y"), value_type=int
                        ),
                        "goal_cell_x": ParameterValue(
                            LaunchConfiguration("goal_cell_x"), value_type=int
                        ),
                        "goal_cell_y": ParameterValue(
                            LaunchConfiguration("goal_cell_y"), value_type=int
                        ),
                    }
                ],
            ),
        ]
    )
