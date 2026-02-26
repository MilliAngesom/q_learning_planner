import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("q_learning_planner")
    default_map_path = os.path.join(package_share, "config", "sample_grid_map.json")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map_config_path",
                default_value=default_map_path,
                description="Path to JSON map config with occupied/free grid cells.",
            ),
            DeclareLaunchArgument(
                "output_q_table_path",
                default_value="/tmp/q_learning_q_tables.npz",
                description="Path where trained Q-table model will be saved.",
            ),
            DeclareLaunchArgument("episodes_per_goal", default_value="300"),
            DeclareLaunchArgument("max_steps_per_episode", default_value="200"),
            DeclareLaunchArgument("alpha", default_value="0.2"),
            DeclareLaunchArgument("gamma", default_value="0.95"),
            DeclareLaunchArgument("epsilon_start", default_value="1.0"),
            DeclareLaunchArgument("epsilon_min", default_value="0.05"),
            DeclareLaunchArgument("epsilon_decay", default_value="0.995"),
            DeclareLaunchArgument("step_penalty", default_value="-0.2"),
            DeclareLaunchArgument("obstacle_penalty", default_value="-5.0"),
            DeclareLaunchArgument("goal_reward", default_value="100.0"),
            DeclareLaunchArgument("distance_reward_scale", default_value="1.0"),
            DeclareLaunchArgument("seed", default_value="42"),
            DeclareLaunchArgument("log_every_n_goals", default_value="10"),
            Node(
                package="q_learning_planner",
                executable="train_q_table",
                output="screen",
                parameters=[
                    {
                        "map_config_path": LaunchConfiguration("map_config_path"),
                        "output_q_table_path": LaunchConfiguration("output_q_table_path"),
                        "episodes_per_goal": ParameterValue(
                            LaunchConfiguration("episodes_per_goal"), value_type=int
                        ),
                        "max_steps_per_episode": ParameterValue(
                            LaunchConfiguration("max_steps_per_episode"),
                            value_type=int,
                        ),
                        "alpha": ParameterValue(
                            LaunchConfiguration("alpha"), value_type=float
                        ),
                        "gamma": ParameterValue(
                            LaunchConfiguration("gamma"), value_type=float
                        ),
                        "epsilon_start": ParameterValue(
                            LaunchConfiguration("epsilon_start"), value_type=float
                        ),
                        "epsilon_min": ParameterValue(
                            LaunchConfiguration("epsilon_min"), value_type=float
                        ),
                        "epsilon_decay": ParameterValue(
                            LaunchConfiguration("epsilon_decay"), value_type=float
                        ),
                        "step_penalty": ParameterValue(
                            LaunchConfiguration("step_penalty"), value_type=float
                        ),
                        "obstacle_penalty": ParameterValue(
                            LaunchConfiguration("obstacle_penalty"), value_type=float
                        ),
                        "goal_reward": ParameterValue(
                            LaunchConfiguration("goal_reward"), value_type=float
                        ),
                        "distance_reward_scale": ParameterValue(
                            LaunchConfiguration("distance_reward_scale"),
                            value_type=float,
                        ),
                        "seed": ParameterValue(
                            LaunchConfiguration("seed"), value_type=int
                        ),
                        "log_every_n_goals": ParameterValue(
                            LaunchConfiguration("log_every_n_goals"), value_type=int
                        ),
                    }
                ],
            ),
        ]
    )
