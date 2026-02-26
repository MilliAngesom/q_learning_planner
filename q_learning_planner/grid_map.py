from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

Cell = Tuple[int, int]


@dataclass(frozen=True)
class GridMap:
    width: int
    height: int
    resolution: float
    origin: Tuple[float, float]
    occupancy: np.ndarray

    @property
    def state_count(self) -> int:
        return self.width * self.height

    @staticmethod
    def from_json(path: str | Path) -> "GridMap":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as config_file:
            data = json.load(config_file)

        width = int(data["width"])
        height = int(data["height"])
        resolution = float(data.get("resolution", 1.0))
        origin_values = data.get("origin", [0.0, 0.0])
        origin = (float(origin_values[0]), float(origin_values[1]))

        occupancy = np.zeros((height, width), dtype=bool)
        for item in data.get("occupied_cells", []):
            if len(item) != 2:
                raise ValueError(f"Invalid occupied cell entry: {item}")
            x = int(item[0])
            y = int(item[1])
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Occupied cell out of bounds: {(x, y)}")
            occupancy[y, x] = True

        return GridMap(
            width=width,
            height=height,
            resolution=resolution,
            origin=origin,
            occupancy=occupancy,
        )

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def is_occupied(self, cell: Cell) -> bool:
        if not self.in_bounds(cell):
            return True
        x, y = cell
        return bool(self.occupancy[y, x])

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and not self.is_occupied(cell)

    def free_cells(self) -> List[Cell]:
        cells: List[Cell] = []
        for y in range(self.height):
            for x in range(self.width):
                if not self.occupancy[y, x]:
                    cells.append((x, y))
        return cells

    def cell_to_state(self, cell: Cell) -> int:
        if not self.in_bounds(cell):
            raise ValueError(f"Cell out of bounds: {cell}")
        x, y = cell
        return y * self.width + x

    def state_to_cell(self, state_index: int) -> Cell:
        if state_index < 0 or state_index >= self.state_count:
            raise ValueError(f"Invalid state index: {state_index}")
        x = state_index % self.width
        y = state_index // self.width
        return (x, y)

    def world_to_cell(self, x_world: float, y_world: float) -> Cell:
        x = int(math.floor((x_world - self.origin[0]) / self.resolution))
        y = int(math.floor((y_world - self.origin[1]) / self.resolution))
        return (x, y)

    def cell_center(self, cell: Cell) -> Tuple[float, float]:
        if not self.in_bounds(cell):
            raise ValueError(f"Cell out of bounds: {cell}")
        x, y = cell
        x_world = self.origin[0] + (x + 0.5) * self.resolution
        y_world = self.origin[1] + (y + 0.5) * self.resolution
        return (x_world, y_world)
