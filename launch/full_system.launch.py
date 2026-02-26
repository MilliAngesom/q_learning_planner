import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("q_learning_planner")
    default_world = os.path.join(package_share, "worlds", "q_learning_world.sdf")
    default_map_path = os.path.join(package_share, "config", "generated_grid_map.json")
    sim_launch_file = os.path.join(package_share, "launch", "sim_with_planner.launch.py")

    sim_launch_args = {
        "world": LaunchConfiguration("world"),
        "model_path": LaunchConfiguration("model_path"),
        "map_config_path": LaunchConfiguration("map_config_path"),
        "rviz": LaunchConfiguration("rviz"),
    }

    train_node = Node(
        package="q_learning_planner",
        executable="train_q_table",
        output="screen",
        condition=IfCondition(LaunchConfiguration("train_rl")),
        parameters=[
            {
                "map_config_path": LaunchConfiguration("map_config_path"),
                "output_q_table_path": LaunchConfiguration("model_path"),
                "episodes_per_goal": ParameterValue(
                    LaunchConfiguration("episodes_per_goal"), value_type=int
                ),
                "max_steps_per_episode": ParameterValue(
                    LaunchConfiguration("max_steps_per_episode"),
                    value_type=int,
                ),
                "alpha": ParameterValue(LaunchConfiguration("alpha"), value_type=float),
                "gamma": ParameterValue(LaunchConfiguration("gamma"), value_type=float),
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
                    LaunchConfiguration("distance_reward_scale"), value_type=float
                ),
                "desired_clearance_cells": ParameterValue(
                    LaunchConfiguration("desired_clearance_cells"), value_type=int
                ),
                "clearance_penalty_weight": ParameterValue(
                    LaunchConfiguration("clearance_penalty_weight"),
                    value_type=float,
                ),
                "seed": ParameterValue(LaunchConfiguration("seed"), value_type=int),
                "log_every_n_goals": ParameterValue(
                    LaunchConfiguration("log_every_n_goals"), value_type=int
                ),
            }
        ],
    )

    sim_direct = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch_file),
        launch_arguments=sim_launch_args.items(),
        condition=UnlessCondition(LaunchConfiguration("train_rl")),
    )

    sim_after_train = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch_file),
        launch_arguments=sim_launch_args.items(),
    )

    sim_after_training_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=train_node,
            on_exit=[sim_after_train],
        ),
        condition=IfCondition(LaunchConfiguration("train_rl")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "train_rl",
                default_value="true",
                description="If true, train Q-table before starting simulation stack.",
            ),
            DeclareLaunchArgument(
                "world",
                default_value=default_world,
                description="Gazebo world file path.",
            ),
            DeclareLaunchArgument(
                "map_config_path",
                default_value=default_map_path,
                description="Path to generated occupancy map JSON.",
            ),
            DeclareLaunchArgument(
                "model_path",
                default_value="/tmp/q_learning_q_tables.npz",
                description="Q-table output path (training) and input path (planner).",
            ),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("episodes_per_goal", default_value="220"),
            DeclareLaunchArgument("max_steps_per_episode", default_value="220"),
            DeclareLaunchArgument("alpha", default_value="0.2"),
            DeclareLaunchArgument("gamma", default_value="0.95"),
            DeclareLaunchArgument("epsilon_start", default_value="1.0"),
            DeclareLaunchArgument("epsilon_min", default_value="0.05"),
            DeclareLaunchArgument("epsilon_decay", default_value="0.996"),
            DeclareLaunchArgument("step_penalty", default_value="-0.25"),
            DeclareLaunchArgument("obstacle_penalty", default_value="-6.0"),
            DeclareLaunchArgument("goal_reward", default_value="120.0"),
            DeclareLaunchArgument("distance_reward_scale", default_value="1.25"),
            DeclareLaunchArgument("desired_clearance_cells", default_value="2"),
            DeclareLaunchArgument("clearance_penalty_weight", default_value="2.5"),
            DeclareLaunchArgument("seed", default_value="42"),
            DeclareLaunchArgument("log_every_n_goals", default_value="10"),
            train_node,
            sim_after_training_handler,
            sim_direct,
        ]
    )
