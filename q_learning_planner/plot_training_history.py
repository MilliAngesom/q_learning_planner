from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/q_learning_planner_matplotlib")

from .q_learning_core import TrainingHistory, load_training_history


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)

    window = max(1, min(window, values.size))
    if window == 1:
        return values.astype(np.float32)

    values = values.astype(np.float64, copy=False)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    rolling = (cumulative[window:] - cumulative[:-window]) / float(window)
    prefix = np.array(
        [np.mean(values[:index], dtype=np.float64) for index in range(1, window)],
        dtype=np.float64,
    )
    return np.concatenate((prefix, rolling)).astype(np.float32)


def _build_title(history: TrainingHistory, history_path: Path) -> str:
    return (
        f"Training curves: {history_path.name}\n"
        f"{history.episode_count} episodes across {history.total_goals} goals "
        f"(seed={history.seed}, episodes/goal={history.episodes_per_goal})"
    )


def _load_plotting_modules(show_plot: bool) -> tuple[Any, Any]:
    try:
        import matplotlib
        from matplotlib import rcsetup

        if show_plot:
            backend = matplotlib.get_backend()
            interactive_backends = {
                name.lower() for name in rcsetup.interactive_bk
            }
            if backend.lower() not in interactive_backends:
                for candidate in ("TkAgg", "Qt5Agg", "QtAgg", "GTK3Agg"):
                    try:
                        matplotlib.use(candidate, force=True)
                        backend = matplotlib.get_backend()
                        if backend.lower() in interactive_backends:
                            break
                    except Exception:
                        continue
        else:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install python3-matplotlib first."
        ) from exc

    backend = matplotlib.get_backend()
    non_interactive_backends = {
        name.lower() for name in rcsetup.non_interactive_bk
    }
    if show_plot and backend.lower() in non_interactive_backends:
        raise SystemExit(
            f"No interactive Matplotlib backend is available (current: {backend}). "
            "Run this from a desktop session, or use --output --no-show."
        )

    return matplotlib, plt


def _create_plot(
    history: TrainingHistory,
    history_path: Path,
    window: int,
    title: str | None,
    show_goal_boundaries: bool,
    plt: Any,
) -> Any:
    if history.episode_count == 0:
        raise ValueError("Training history is empty.")

    episodes = history.global_episode_indices.astype(np.int32)
    rewards = history.total_rewards.astype(np.float32)
    steps = history.steps.astype(np.float32)
    successes = history.successes.astype(np.float32) * 100.0
    epsilons = history.epsilons.astype(np.float32)
    effective_window = max(1, min(window, history.episode_count))

    reward_avg = _moving_average(rewards, effective_window)
    steps_avg = _moving_average(steps, effective_window)
    success_avg = _moving_average(successes, effective_window)

    fig, axes = plt.subplots(
        4, 1, figsize=(13, 12), sharex=True, constrained_layout=True
    )
    colors = {
        "reward": "#1f78b4",
        "steps": "#ff7f00",
        "success": "#33a02c",
        "epsilon": "#6a3d9a",
        "boundary": "#d9d9d9",
    }

    axes[0].plot(episodes, rewards, color=colors["reward"], alpha=0.18, linewidth=0.9)
    axes[0].plot(
        episodes,
        reward_avg,
        color=colors["reward"],
        linewidth=1.8,
        label=f"Reward ({effective_window}-episode avg)",
    )
    axes[0].set_ylabel("Reward")
    axes[0].legend(loc="upper right")

    axes[1].plot(episodes, steps, color=colors["steps"], alpha=0.18, linewidth=0.9)
    axes[1].plot(
        episodes,
        steps_avg,
        color=colors["steps"],
        linewidth=1.8,
        label=f"Steps ({effective_window}-episode avg)",
    )
    axes[1].set_ylabel("Steps")
    axes[1].legend(loc="upper right")

    axes[2].plot(
        episodes,
        successes,
        color=colors["success"],
        alpha=0.12,
        linewidth=0.9,
    )
    axes[2].plot(
        episodes,
        success_avg,
        color=colors["success"],
        linewidth=1.8,
        label=f"Success ({effective_window}-episode avg)",
    )
    axes[2].set_ylabel("Success %")
    axes[2].set_ylim(-2.0, 102.0)
    axes[2].legend(loc="upper right")

    axes[3].plot(episodes, epsilons, color=colors["epsilon"], linewidth=1.3)
    axes[3].set_ylabel("Epsilon")
    axes[3].set_xlabel("Global episode")

    if show_goal_boundaries and history.episodes_per_goal > 0:
        for boundary in range(
            history.episodes_per_goal, history.episode_count, history.episodes_per_goal
        ):
            for axis in axes:
                axis.axvline(
                    boundary,
                    color=colors["boundary"],
                    linewidth=0.7,
                    linestyle="--",
                    zorder=0,
                )

    for axis in axes:
        axis.grid(True, alpha=0.25)

    fig.suptitle(title if title is not None else _build_title(history, history_path))
    return fig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot full Q-learning training curves from a saved history file."
    )
    parser.add_argument(
        "history_path",
        help="Path to the saved training history .npz file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output image path if you also want to save the plot.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=200,
        help="Moving-average window in episodes for the smoothed curves.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Saved figure DPI.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title override.",
    )
    parser.add_argument(
        "--goal-boundaries",
        action="store_true",
        help="Draw dashed vertical lines at goal-to-goal training boundaries.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive plot window.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    history_path = Path(args.history_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    show_plot = not bool(args.no_show)
    if not show_plot and output_path is None:
        parser.error("Nothing to do. Omit --no-show or pass --output.")

    _, plt = _load_plotting_modules(show_plot=show_plot)
    history = load_training_history(history_path)
    window = max(1, int(args.window))

    fig = _create_plot(
        history=history,
        history_path=history_path,
        window=window,
        title=args.title,
        show_goal_boundaries=bool(args.goal_boundaries),
        plt=plt,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=max(72, int(args.dpi)))
        print(f"Saved training curves to: {output_path}")

    if show_plot:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
