from __future__ import annotations

from pathlib import Path

import rclpy
from rclpy.node import Node

from .grid_map import GridMap
from .q_learning_core import QLearningConfig, QLearningTrainer


class QTableTrainerNode(Node):
    def __init__(self) -> None:
        super().__init__("q_table_trainer")

        self.declare_parameter("map_config_path", "")
        self.declare_parameter("output_q_table_path", "/tmp/q_learning_q_tables.npz")
        self.declare_parameter("episodes_per_goal", 400)
        self.declare_parameter("max_steps_per_episode", 200)
        self.declare_parameter("alpha", 0.2)
        self.declare_parameter("gamma", 0.95)
        self.declare_parameter("epsilon_start", 1.0)
        self.declare_parameter("epsilon_min", 0.05)
        self.declare_parameter("epsilon_decay", 0.995)
        self.declare_parameter("step_penalty", -0.2)
        self.declare_parameter("obstacle_penalty", -5.0)
        self.declare_parameter("goal_reward", 100.0)
        self.declare_parameter("distance_reward_scale", 1.0)
        self.declare_parameter("seed", 42)
        self.declare_parameter("log_every_n_goals", 10)

    def run(self) -> None:
        map_config_path = str(self.get_parameter("map_config_path").value).strip()
        if not map_config_path:
            raise ValueError(
                "Parameter 'map_config_path' is required and must point to a JSON map file."
            )

        output_q_table_path = str(self.get_parameter("output_q_table_path").value)
        episodes_per_goal = int(self.get_parameter("episodes_per_goal").value)
        max_steps = int(self.get_parameter("max_steps_per_episode").value)
        seed = int(self.get_parameter("seed").value)
        log_every = max(1, int(self.get_parameter("log_every_n_goals").value))

        config = QLearningConfig(
            alpha=float(self.get_parameter("alpha").value),
            gamma=float(self.get_parameter("gamma").value),
            epsilon_start=float(self.get_parameter("epsilon_start").value),
            epsilon_min=float(self.get_parameter("epsilon_min").value),
            epsilon_decay=float(self.get_parameter("epsilon_decay").value),
            step_penalty=float(self.get_parameter("step_penalty").value),
            obstacle_penalty=float(self.get_parameter("obstacle_penalty").value),
            goal_reward=float(self.get_parameter("goal_reward").value),
            distance_reward_scale=float(
                self.get_parameter("distance_reward_scale").value
            ),
        )

        self.get_logger().info(f"Loading map config: {map_config_path}")
        grid_map = GridMap.from_json(map_config_path)
        self.get_logger().info(
            f"Map loaded ({grid_map.width}x{grid_map.height}, "
            f"free cells={len(grid_map.free_cells())})."
        )

        trainer = QLearningTrainer(grid_map=grid_map, config=config)

        def on_progress(done_goals: int, total_goals: int, success_rate: float) -> None:
            if done_goals % log_every == 0 or done_goals == total_goals:
                self.get_logger().info(
                    "Training progress: "
                    f"{done_goals}/{total_goals} goals, "
                    f"last-goal success={success_rate:.2f}"
                )

        self.get_logger().info(
            "Starting training with "
            f"episodes_per_goal={episodes_per_goal}, max_steps={max_steps}, seed={seed}"
        )
        model = trainer.train_all_goals(
            episodes_per_goal=episodes_per_goal,
            max_steps_per_episode=max_steps,
            rng_seed=seed,
            progress_callback=on_progress,
        )

        Path(output_q_table_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(output_q_table_path)
        self.get_logger().info(f"Saved Q-table model to: {output_q_table_path}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = QTableTrainerNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
