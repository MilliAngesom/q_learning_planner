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
EpisodeCallback = Optional[Callable[["TrainingEpisodeMetrics"], None]]


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
    desired_clearance_cells: int = 3
    clearance_penalty_weight: float = 4.0


@dataclass
class TrainingEpisodeMetrics:
    goal_index: int
    goal_cell: Cell
    episode_index: int
    global_episode_index: int
    epsilon: float
    success: bool
    steps: int
    total_reward: float


@dataclass
class TrainingHistory:
    goal_indices: np.ndarray
    goal_cells: np.ndarray
    episode_indices: np.ndarray
    global_episode_indices: np.ndarray
    epsilons: np.ndarray
    successes: np.ndarray
    steps: np.ndarray
    total_rewards: np.ndarray
    episodes_per_goal: int
    max_steps_per_episode: int
    seed: int
    total_goals: int

    @property
    def episode_count(self) -> int:
        return int(self.global_episode_indices.size)

    @classmethod
    def from_records(
        cls,
        records: List[TrainingEpisodeMetrics],
        episodes_per_goal: int,
        max_steps_per_episode: int,
        seed: int,
        total_goals: int,
    ) -> "TrainingHistory":
        if not records:
            empty_int = np.array([], dtype=np.int32)
            empty_float = np.array([], dtype=np.float32)
            empty_bool = np.array([], dtype=bool)
            return cls(
                goal_indices=empty_int,
                goal_cells=np.empty((0, 2), dtype=np.int32),
                episode_indices=empty_int,
                global_episode_indices=empty_int,
                epsilons=empty_float,
                successes=empty_bool,
                steps=empty_int,
                total_rewards=empty_float,
                episodes_per_goal=episodes_per_goal,
                max_steps_per_episode=max_steps_per_episode,
                seed=seed,
                total_goals=total_goals,
            )

        return cls(
            goal_indices=np.array(
                [record.goal_index for record in records], dtype=np.int32
            ),
            goal_cells=np.array(
                [record.goal_cell for record in records], dtype=np.int32
            ),
            episode_indices=np.array(
                [record.episode_index for record in records], dtype=np.int32
            ),
            global_episode_indices=np.array(
                [record.global_episode_index for record in records], dtype=np.int32
            ),
            epsilons=np.array(
                [record.epsilon for record in records], dtype=np.float32
            ),
            successes=np.array([record.success for record in records], dtype=bool),
            steps=np.array([record.steps for record in records], dtype=np.int32),
            total_rewards=np.array(
                [record.total_reward for record in records], dtype=np.float32
            ),
            episodes_per_goal=episodes_per_goal,
            max_steps_per_episode=max_steps_per_episode,
            seed=seed,
            total_goals=total_goals,
        )

    def save(self, output_path: str | Path) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            goal_indices=self.goal_indices.astype(np.int32),
            goal_cells=self.goal_cells.astype(np.int32),
            episode_indices=self.episode_indices.astype(np.int32),
            global_episode_indices=self.global_episode_indices.astype(np.int32),
            epsilons=self.epsilons.astype(np.float32),
            successes=self.successes.astype(bool),
            steps=self.steps.astype(np.int32),
            total_rewards=self.total_rewards.astype(np.float32),
            episodes_per_goal=np.array([self.episodes_per_goal], dtype=np.int32),
            max_steps_per_episode=np.array(
                [self.max_steps_per_episode], dtype=np.int32
            ),
            seed=np.array([self.seed], dtype=np.int32),
            total_goals=np.array([self.total_goals], dtype=np.int32),
        )


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


def load_training_history(history_path: str | Path) -> TrainingHistory:
    history_file = Path(history_path)
    if not history_file.exists():
        raise FileNotFoundError(f"Training history not found: {history_file}")

    loaded = np.load(history_file, allow_pickle=False)
    return TrainingHistory(
        goal_indices=loaded["goal_indices"].astype(np.int32),
        goal_cells=loaded["goal_cells"].astype(np.int32),
        episode_indices=loaded["episode_indices"].astype(np.int32),
        global_episode_indices=loaded["global_episode_indices"].astype(np.int32),
        epsilons=loaded["epsilons"].astype(np.float32),
        successes=loaded["successes"].astype(bool),
        steps=loaded["steps"].astype(np.int32),
        total_rewards=loaded["total_rewards"].astype(np.float32),
        episodes_per_goal=int(loaded["episodes_per_goal"][0]),
        max_steps_per_episode=int(loaded["max_steps_per_episode"][0]),
        seed=int(loaded["seed"][0]),
        total_goals=int(loaded["total_goals"][0]),
    )


@dataclass
class PlanningResult:
    path: List[Cell]
    used_fallback: bool


class QLearningTrainer:
    def __init__(self, grid_map: GridMap, config: Optional[QLearningConfig] = None):
        self._grid_map = grid_map
        self._config = config if config is not None else QLearningConfig()
        self._obstacle_distance_map = self._build_obstacle_distance_map()

    def train_all_goals(
        self,
        episodes_per_goal: int,
        max_steps_per_episode: int,
        rng_seed: int,
        progress_callback: ProgressCallback = None,
        episode_callback: EpisodeCallback = None,
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
        global_episode_index = 0

        for goal_index, goal_cell in enumerate(free_cells):
            q_table = q_tables[goal_index]
            epsilon = self._config.epsilon_start
            success_count = 0

            for episode_index in range(episodes_per_goal):
                start_cell = goal_cell
                while start_cell == goal_cell:
                    random_index = int(rng.integers(0, len(free_cells)))
                    start_cell = free_cells[random_index]

                episode_epsilon = epsilon
                current_cell = start_cell
                total_reward = 0.0
                steps_taken = 0
                succeeded = False

                for step_index in range(max_steps_per_episode):
                    current_state = self._grid_map.cell_to_state(current_cell)
                    action = self._select_action(q_table, current_state, epsilon, rng)
                    next_cell, reward, done = self._step(
                        current_cell, action, goal_cell
                    )
                    next_state = self._grid_map.cell_to_state(next_cell)
                    total_reward += reward
                    steps_taken = step_index + 1

                    next_value = 0.0 if done else float(np.max(q_table[next_state]))
                    td_target = reward + self._config.gamma * next_value
                    td_error = td_target - q_table[current_state, action]
                    q_table[current_state, action] += self._config.alpha * td_error

                    current_cell = next_cell
                    if done:
                        success_count += 1
                        succeeded = True
                        break

                global_episode_index += 1
                if episode_callback is not None:
                    episode_callback(
                        TrainingEpisodeMetrics(
                            goal_index=goal_index,
                            goal_cell=goal_cell,
                            episode_index=episode_index,
                            global_episode_index=global_episode_index,
                            epsilon=float(episode_epsilon),
                            success=succeeded,
                            steps=steps_taken,
                            total_reward=float(total_reward),
                        )
                    )

                epsilon = max(
                    self._config.epsilon_min, epsilon * self._config.epsilon_decay
                )

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
        clearance = self._obstacle_distance_map[candidate[1]][candidate[0]]
        shortage = max(0, self._config.desired_clearance_cells - clearance)
        clearance_penalty = self._config.clearance_penalty_weight * float(
            shortage * shortage
        )
        reward = self._config.step_penalty + distance_bonus - clearance_penalty
        return candidate, reward, False

    @staticmethod
    def _manhattan(first: Cell, second: Cell) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def _build_obstacle_distance_map(self) -> np.ndarray:
        inf_dist = self._grid_map.width + self._grid_map.height + 1
        distance_map = np.full(
            (self._grid_map.height, self._grid_map.width), inf_dist, dtype=np.int32
        )
        queue: deque[Cell] = deque()

        for y in range(self._grid_map.height):
            for x in range(self._grid_map.width):
                if self._grid_map.occupancy[y, x]:
                    distance_map[y, x] = 0
                    queue.append((x, y))

        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
        while queue:
            current = queue.popleft()
            current_dist = int(distance_map[current[1], current[0]])
            for dx, dy in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if not self._grid_map.in_bounds(nxt):
                    continue
                nxt_dist = current_dist + 1
                if nxt_dist >= int(distance_map[nxt[1], nxt[0]]):
                    continue
                distance_map[nxt[1], nxt[0]] = nxt_dist
                queue.append(nxt)

        return distance_map


class QTablePlanner:
    def __init__(self, model: QTableModel):
        self._model = model
        self._goal_to_index: Dict[Cell, int] = {
            (int(cell[0]), int(cell[1])): idx
            for idx, cell in enumerate(model.free_cells.tolist())
        }
        self._obstacle_distance_map = self._build_obstacle_distance_map()

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
            cleaned_path = self._prune_cycles(learned_path)
            return PlanningResult(path=cleaned_path, used_fallback=False)

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
        if max_steps <= 0:
            return []
        revisit_limits = (0, 1, 2, 4, 8, 999_999)
        for revisit_limit in revisit_limits:
            path: List[Cell] = [start_cell]
            current_cell = start_cell
            visited_count: Dict[Cell, int] = {start_cell: 1}

            for _ in range(max_steps):
                if current_cell == goal_cell:
                    return path

                next_cell = self._choose_next_cell(
                    current_cell=current_cell,
                    goal_cell=goal_cell,
                    q_table=q_table,
                    path=path,
                    visited_count=visited_count,
                    revisit_limit=revisit_limit,
                )
                if next_cell is None:
                    break

                current_cell = next_cell
                path.append(current_cell)
                visited_count[current_cell] = visited_count.get(current_cell, 0) + 1

                # Early stop for oscillation patterns.
                if len(path) >= 12 and len(set(path[-12:])) <= 4:
                    break

            if path and path[-1] == goal_cell:
                return path

        return []

    def _choose_next_cell(
        self,
        current_cell: Cell,
        goal_cell: Cell,
        q_table: np.ndarray,
        path: List[Cell],
        visited_count: Dict[Cell, int],
        revisit_limit: int,
    ) -> Optional[Cell]:
        state = self._model.grid_map.cell_to_state(current_cell)
        action_order = np.argsort(q_table[state])[::-1]
        candidate_groups: List[Tuple[float, List[Cell]]] = []
        best_q: Optional[float] = None
        for action in action_order:
            q_value = float(q_table[state, int(action)])
            next_cell = self._candidate_cell(current_cell, int(action))
            if not self._model.grid_map.is_free(next_cell):
                continue
            if best_q is None:
                best_q = q_value
                candidate_groups.append((best_q, [next_cell]))
                continue
            if abs(q_value - best_q) < 1e-6:
                candidate_groups[0][1].append(next_cell)
            else:
                candidate_groups.append((q_value, [next_cell]))

        if not candidate_groups:
            return None

        previous_cell = path[-2] if len(path) >= 2 else None
        fallback: Optional[Cell] = None
        for _, same_q_cells in candidate_groups:
            ranked_cells = sorted(
                same_q_cells,
                key=lambda c: (
                    visited_count.get(c, 0),
                    self._manhattan(c, goal_cell),
                    -float(self._obstacle_distance_map[c[1], c[0]]),
                ),
            )
            for cell in ranked_cells:
                if fallback is None:
                    fallback = cell
                if visited_count.get(cell, 0) > revisit_limit:
                    continue
                if previous_cell is not None and cell == previous_cell and revisit_limit <= 2:
                    continue
                return cell

        return fallback

    @staticmethod
    def _prune_cycles(path: List[Cell]) -> List[Cell]:
        if len(path) <= 2:
            return path

        pruned: List[Cell] = []
        index_by_cell: Dict[Cell, int] = {}
        for cell in path:
            if cell in index_by_cell:
                keep_until = index_by_cell[cell]
                pruned = pruned[: keep_until + 1]
                index_by_cell = {c: i for i, c in enumerate(pruned)}
                continue
            index_by_cell[cell] = len(pruned)
            pruned.append(cell)
        return pruned

    def _build_obstacle_distance_map(self) -> np.ndarray:
        inf_dist = self._model.grid_map.width + self._model.grid_map.height + 1
        distance_map = np.full(
            (self._model.grid_map.height, self._model.grid_map.width),
            float(inf_dist),
            dtype=np.float32,
        )
        queue: deque[Cell] = deque()

        for y in range(self._model.grid_map.height):
            for x in range(self._model.grid_map.width):
                if self._model.grid_map.occupancy[y, x]:
                    distance_map[y, x] = 0.0
                    queue.append((x, y))

        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
        while queue:
            current = queue.popleft()
            current_dist = float(distance_map[current[1], current[0]])
            for dx, dy in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if not self._model.grid_map.in_bounds(nxt):
                    continue
                nxt_dist = current_dist + 1.0
                if nxt_dist >= float(distance_map[nxt[1], nxt[0]]):
                    continue
                distance_map[nxt[1], nxt[0]] = nxt_dist
                queue.append(nxt)

        return distance_map

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

    @staticmethod
    def _manhattan(first: Cell, second: Cell) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])
