from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .grid_map import Cell, GridMap

ACTION_DELTAS: Tuple[Cell, ...] = (
    (1, 0),   # right
    (-1, 0),  # left
    (0, 1),   # up
    (0, -1),  # down
)

ProgressCallback = Optional[Callable[[int, int, float], None]]


@dataclass
class QLearningConfig:
    alpha: float = 0.2
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    step_penalty: float = -0.2
    obstacle_penalty: float = -5.0
    goal_reward: float = 100.0
    distance_reward_scale: float = 1.0


@dataclass
class QTableModel:
    grid_map: GridMap
    q_tables: np.ndarray
    free_cells: np.ndarray

    def save(self, output_path: str | Path) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            q_tables=self.q_tables.astype(np.float32),
            free_cells=self.free_cells.astype(np.int32),
            width=np.array([self.grid_map.width], dtype=np.int32),
            height=np.array([self.grid_map.height], dtype=np.int32),
            resolution=np.array([self.grid_map.resolution], dtype=np.float32),
            origin=np.array(self.grid_map.origin, dtype=np.float32),
            occupancy=self.grid_map.occupancy.astype(np.uint8),
        )


def load_q_table_model(model_path: str | Path) -> QTableModel:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Q-table model not found: {model_file}")

    loaded = np.load(model_file, allow_pickle=False)
    width = int(loaded["width"][0])
    height = int(loaded["height"][0])
    resolution = float(loaded["resolution"][0])
    origin = tuple(float(v) for v in loaded["origin"])
    occupancy = loaded["occupancy"].astype(bool)
    q_tables = loaded["q_tables"].astype(np.float32)
    free_cells = loaded["free_cells"].astype(np.int32)

    grid_map = GridMap(
        width=width,
        height=height,
        resolution=resolution,
        origin=(origin[0], origin[1]),
        occupancy=occupancy,
    )

    return QTableModel(grid_map=grid_map, q_tables=q_tables, free_cells=free_cells)


@dataclass
class PlanningResult:
    path: List[Cell]
    used_fallback: bool


class QLearningTrainer:
    def __init__(self, grid_map: GridMap, config: Optional[QLearningConfig] = None):
        self._grid_map = grid_map
        self._config = config if config is not None else QLearningConfig()

    def train_all_goals(
        self,
        episodes_per_goal: int,
        max_steps_per_episode: int,
        rng_seed: int,
        progress_callback: ProgressCallback = None,
    ) -> QTableModel:
        free_cells = self._grid_map.free_cells()
        if len(free_cells) < 2:
            raise ValueError("Map must contain at least 2 free cells for training.")

        total_goals = len(free_cells)
        total_states = self._grid_map.state_count
        total_actions = len(ACTION_DELTAS)
        q_tables = np.zeros(
            (total_goals, total_states, total_actions), dtype=np.float32
        )
        rng = np.random.default_rng(rng_seed)

        for goal_index, goal_cell in enumerate(free_cells):
            q_table = q_tables[goal_index]
            epsilon = self._config.epsilon_start
            success_count = 0

            for _ in range(episodes_per_goal):
                start_cell = goal_cell
                while start_cell == goal_cell:
                    random_index = int(rng.integers(0, len(free_cells)))
                    start_cell = free_cells[random_index]

                current_cell = start_cell
                for _ in range(max_steps_per_episode):
                    current_state = self._grid_map.cell_to_state(current_cell)
                    action = self._select_action(q_table, current_state, epsilon, rng)
                    next_cell, reward, done = self._step(current_cell, action, goal_cell)
                    next_state = self._grid_map.cell_to_state(next_cell)

                    next_value = 0.0 if done else float(np.max(q_table[next_state]))
                    td_target = reward + self._config.gamma * next_value
                    td_error = td_target - q_table[current_state, action]
                    q_table[current_state, action] += self._config.alpha * td_error

                    current_cell = next_cell
                    if done:
                        success_count += 1
                        break

                epsilon = max(self._config.epsilon_min, epsilon * self._config.epsilon_decay)

            if progress_callback is not None:
                success_rate = success_count / float(episodes_per_goal)
                progress_callback(goal_index + 1, total_goals, success_rate)

        return QTableModel(
            grid_map=self._grid_map,
            q_tables=q_tables,
            free_cells=np.array(free_cells, dtype=np.int32),
        )

    def _select_action(
        self,
        q_table: np.ndarray,
        state_index: int,
        epsilon: float,
        rng: np.random.Generator,
    ) -> int:
        if float(rng.random()) < epsilon:
            return int(rng.integers(0, len(ACTION_DELTAS)))
        return int(np.argmax(q_table[state_index]))

    def _step(self, current: Cell, action: int, goal: Cell) -> Tuple[Cell, float, bool]:
        dx, dy = ACTION_DELTAS[action]
        candidate = (current[0] + dx, current[1] + dy)
        old_dist = self._manhattan(current, goal)

        if not self._grid_map.is_free(candidate):
            reward = self._config.step_penalty + self._config.obstacle_penalty
            return current, reward, False

        if candidate == goal:
            return candidate, self._config.goal_reward, True

        new_dist = self._manhattan(candidate, goal)
        distance_bonus = self._config.distance_reward_scale * (old_dist - new_dist)
        reward = self._config.step_penalty + distance_bonus
        return candidate, reward, False

    @staticmethod
    def _manhattan(first: Cell, second: Cell) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])


class QTablePlanner:
    def __init__(self, model: QTableModel):
        self._model = model
        self._goal_to_index: Dict[Cell, int] = {
            (int(cell[0]), int(cell[1])): idx
            for idx, cell in enumerate(model.free_cells.tolist())
        }

    @property
    def model(self) -> QTableModel:
        return self._model

    def plan(
        self,
        start_cell: Cell,
        goal_cell: Cell,
        max_steps: int,
        allow_fallback: bool,
    ) -> PlanningResult:
        if not self._model.grid_map.is_free(start_cell):
            return PlanningResult(path=[], used_fallback=False)
        if not self._model.grid_map.is_free(goal_cell):
            return PlanningResult(path=[], used_fallback=False)

        if start_cell == goal_cell:
            return PlanningResult(path=[start_cell], used_fallback=False)

        goal_index = self._goal_to_index.get(goal_cell)
        if goal_index is None:
            return PlanningResult(path=[], used_fallback=False)

        q_table = self._model.q_tables[goal_index]
        learned_path = self._plan_with_q_table(start_cell, goal_cell, q_table, max_steps)
        if learned_path and learned_path[-1] == goal_cell:
            return PlanningResult(path=learned_path, used_fallback=False)

        if allow_fallback:
            bfs_path = self._bfs_path(start_cell, goal_cell)
            if bfs_path:
                return PlanningResult(path=bfs_path, used_fallback=True)

        return PlanningResult(path=[], used_fallback=False)

    def _plan_with_q_table(
        self,
        start_cell: Cell,
        goal_cell: Cell,
        q_table: np.ndarray,
        max_steps: int,
    ) -> List[Cell]:
        path: List[Cell] = [start_cell]
        current_cell = start_cell
        previous_cell: Optional[Cell] = None

        for _ in range(max_steps):
            if current_cell == goal_cell:
                return path

            state = self._model.grid_map.cell_to_state(current_cell)
            action_order = np.argsort(q_table[state])[::-1]

            chosen_next: Optional[Cell] = None
            for action in action_order:
                next_cell = self._candidate_cell(current_cell, int(action))
                if not self._model.grid_map.is_free(next_cell):
                    continue
                if previous_cell is not None and next_cell == previous_cell:
                    continue
                chosen_next = next_cell
                break

            if chosen_next is None:
                for action in action_order:
                    next_cell = self._candidate_cell(current_cell, int(action))
                    if self._model.grid_map.is_free(next_cell):
                        chosen_next = next_cell
                        break

            if chosen_next is None:
                return []

            previous_cell = current_cell
            current_cell = chosen_next
            path.append(current_cell)

            if len(path) >= 4 and path[-1] == path[-3] and path[-2] == path[-4]:
                return []

        return []

    def _bfs_path(self, start_cell: Cell, goal_cell: Cell) -> List[Cell]:
        queue: deque[Cell] = deque([start_cell])
        parent: Dict[Cell, Optional[Cell]] = {start_cell: None}

        while queue:
            current = queue.popleft()
            if current == goal_cell:
                break

            for delta in ACTION_DELTAS:
                next_cell = (current[0] + delta[0], current[1] + delta[1])
                if not self._model.grid_map.is_free(next_cell):
                    continue
                if next_cell in parent:
                    continue
                parent[next_cell] = current
                queue.append(next_cell)

        if goal_cell not in parent:
            return []

        path: List[Cell] = []
        cursor: Optional[Cell] = goal_cell
        while cursor is not None:
            path.append(cursor)
            cursor = parent[cursor]

        path.reverse()
        return path

    @staticmethod
    def _candidate_cell(current: Cell, action: int) -> Cell:
        dx, dy = ACTION_DELTAS[action]
        return (current[0] + dx, current[1] + dy)
