from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Obstacle:
    name: str
    x: float
    y: float
    size_x: float
    size_y: float
    height: float


def _resolve_default_layout() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        package_share = get_package_share_directory("q_learning_planner")
        return Path(package_share) / "config" / "environment_layout.json"
    except Exception:
        package_root = Path(__file__).resolve().parents[1]
        return package_root / "config" / "environment_layout.json"


def _cell_center(
    x: int, y: int, resolution: float, origin_x: float, origin_y: float
) -> Tuple[float, float]:
    return (
        origin_x + (x + 0.5) * resolution,
        origin_y + (y + 0.5) * resolution,
    )


def _safe_name(name: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in name.strip())
    safe = safe.strip("_")
    if not safe:
        safe = "obstacle"
    return safe


def _world_dimensions_from_layout(
    layout: Dict, resolution: float
) -> Tuple[float, float, int, int]:
    if "world_size" in layout:
        world_size = layout["world_size"]
        if not isinstance(world_size, list) or len(world_size) != 2:
            raise ValueError("world_size must be [width_m, height_m].")
        world_width = float(world_size[0])
        world_height = float(world_size[1])
    elif "world_width" in layout and "world_height" in layout:
        world_width = float(layout["world_width"])
        world_height = float(layout["world_height"])
    elif "width" in layout and "height" in layout:
        width_cells = int(layout["width"])
        height_cells = int(layout["height"])
        if width_cells <= 0 or height_cells <= 0:
            raise ValueError("width and height must be > 0.")
        world_width = width_cells * resolution
        world_height = height_cells * resolution
    else:
        raise ValueError(
            "Layout must define either world_size/world_width+world_height "
            "or width+height."
        )

    if world_width <= 0.0 or world_height <= 0.0:
        raise ValueError("world dimensions must be > 0.")

    width_f = world_width / resolution
    height_f = world_height / resolution
    width = int(round(width_f))
    height = int(round(height_f))
    if abs(width_f - width) > 1e-6 or abs(height_f - height) > 1e-6:
        raise ValueError(
            "world_size must be divisible by resolution so grid dimensions are exact."
        )
    return world_width, world_height, width, height


def _parse_robot_start(
    layout: Dict, resolution: float, origin_x: float, origin_y: float
) -> Tuple[float, float]:
    if "robot_start" in layout:
        robot_start = layout["robot_start"]
        if not isinstance(robot_start, dict):
            raise ValueError("robot_start must be an object with x/y.")
        return (float(robot_start["x"]), float(robot_start["y"]))
    if "robot_start_xy" in layout:
        robot_start_xy = layout["robot_start_xy"]
        if not isinstance(robot_start_xy, list) or len(robot_start_xy) != 2:
            raise ValueError("robot_start_xy must be [x, y].")
        return (float(robot_start_xy[0]), float(robot_start_xy[1]))
    if "robot_start_cell" in layout:
        start_cell = layout["robot_start_cell"]
        if not isinstance(start_cell, list) or len(start_cell) != 2:
            raise ValueError("robot_start_cell must be [x, y].")
        start_x = int(start_cell[0])
        start_y = int(start_cell[1])
        return _cell_center(start_x, start_y, resolution, origin_x, origin_y)
    return (origin_x + 0.5 * resolution, origin_y + 0.5 * resolution)


def _parse_obstacles(
    layout: Dict,
    resolution: float,
    origin_x: float,
    origin_y: float,
    width: int,
    height: int,
) -> List[Obstacle]:
    default_height = float(layout.get("obstacle_height", 0.8))
    if default_height <= 0.0:
        raise ValueError("obstacle_height must be > 0.")

    obstacles: List[Obstacle] = []

    # New schema: explicit metric obstacles, independent from grid resolution.
    if "obstacles" in layout:
        raw_obstacles = layout["obstacles"]
        if not isinstance(raw_obstacles, list):
            raise ValueError("obstacles must be a list.")
        for idx, item in enumerate(raw_obstacles):
            if not isinstance(item, dict):
                raise ValueError(f"Obstacle entry #{idx} must be an object.")
            name = str(item.get("name", f"obs_{idx}"))
            x = float(item["x"])
            y = float(item["y"])
            size_x = float(item.get("size_x", item.get("width", 0.0)))
            size_y = float(item.get("size_y", item.get("height", 0.0)))
            obstacle_height = float(item.get("height", default_height))
            if size_x <= 0.0 or size_y <= 0.0 or obstacle_height <= 0.0:
                raise ValueError(
                    f"Obstacle '{name}' must have positive size_x/size_y/height."
                )
            obstacles.append(
                Obstacle(
                    name=name,
                    x=x,
                    y=y,
                    size_x=size_x,
                    size_y=size_y,
                    height=obstacle_height,
                )
            )
        return obstacles

    # Backward-compatible schema: occupied cells expanded to per-cell obstacles.
    obstacle_ratio = float(layout.get("obstacle_size_ratio", 0.9))
    if obstacle_ratio <= 0.0:
        raise ValueError("obstacle_size_ratio must be > 0.")
    obstacle_size_xy = max(0.01, obstacle_ratio * resolution)
    for item in layout.get("occupied_cells", []):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"Invalid occupied cell entry: {item}")
        cx = int(item[0])
        cy = int(item[1])
        if not (0 <= cx < width and 0 <= cy < height):
            raise ValueError(f"Occupied cell out of bounds: {(cx, cy)}")
        x_world, y_world = _cell_center(cx, cy, resolution, origin_x, origin_y)
        obstacles.append(
            Obstacle(
                name=f"obs_{cx}_{cy}",
                x=x_world,
                y=y_world,
                size_x=obstacle_size_xy,
                size_y=obstacle_size_xy,
                height=default_height,
            )
        )
    return obstacles


def _validate_world_entities(
    origin_x: float,
    origin_y: float,
    world_width: float,
    world_height: float,
    robot_start: Tuple[float, float],
    obstacles: List[Obstacle],
) -> None:
    min_x = origin_x
    min_y = origin_y
    max_x = origin_x + world_width
    max_y = origin_y + world_height

    rx, ry = robot_start
    if not (min_x <= rx <= max_x and min_y <= ry <= max_y):
        raise ValueError("robot_start must be inside world bounds.")

    for obs in obstacles:
        left = obs.x - 0.5 * obs.size_x
        right = obs.x + 0.5 * obs.size_x
        bottom = obs.y - 0.5 * obs.size_y
        top = obs.y + 0.5 * obs.size_y
        if left < min_x or right > max_x or bottom < min_y or top > max_y:
            raise ValueError(
                f"Obstacle '{obs.name}' is outside world bounds. "
                f"Obstacle bounds=({left:.3f},{bottom:.3f})-({right:.3f},{top:.3f}), "
                f"world bounds=({min_x:.3f},{min_y:.3f})-({max_x:.3f},{max_y:.3f})"
            )


def _boxes_overlap(
    min_ax: float,
    min_ay: float,
    max_ax: float,
    max_ay: float,
    min_bx: float,
    min_by: float,
    max_bx: float,
    max_by: float,
) -> bool:
    return (
        min_ax < max_bx
        and max_ax > min_bx
        and min_ay < max_by
        and max_ay > min_by
    )


def _rasterize_obstacles_to_cells(
    obstacles: List[Obstacle],
    resolution: float,
    origin_x: float,
    origin_y: float,
    width: int,
    height: int,
) -> List[List[int]]:
    occupancy = [[False for _ in range(width)] for _ in range(height)]

    for obs in obstacles:
        obs_min_x = obs.x - 0.5 * obs.size_x
        obs_max_x = obs.x + 0.5 * obs.size_x
        obs_min_y = obs.y - 0.5 * obs.size_y
        obs_max_y = obs.y + 0.5 * obs.size_y

        cell_x_min = max(0, int(math.floor((obs_min_x - origin_x) / resolution)))
        cell_x_max = min(
            width - 1, int(math.ceil((obs_max_x - origin_x) / resolution) - 1)
        )
        cell_y_min = max(0, int(math.floor((obs_min_y - origin_y) / resolution)))
        cell_y_max = min(
            height - 1, int(math.ceil((obs_max_y - origin_y) / resolution) - 1)
        )

        if cell_x_min > cell_x_max or cell_y_min > cell_y_max:
            continue

        for cy in range(cell_y_min, cell_y_max + 1):
            cell_min_y = origin_y + cy * resolution
            cell_max_y = cell_min_y + resolution
            for cx in range(cell_x_min, cell_x_max + 1):
                cell_min_x = origin_x + cx * resolution
                cell_max_x = cell_min_x + resolution
                if _boxes_overlap(
                    obs_min_x,
                    obs_min_y,
                    obs_max_x,
                    obs_max_y,
                    cell_min_x,
                    cell_min_y,
                    cell_max_x,
                    cell_max_y,
                ):
                    occupancy[cy][cx] = True

    occupied_cells: List[List[int]] = []
    for y in range(height):
        for x in range(width):
            if occupancy[y][x]:
                occupied_cells.append([x, y])
    return occupied_cells


def _prepare_layout(layout: Dict) -> Dict:
    resolution = float(layout.get("resolution", 1.0))
    if resolution <= 0.0:
        raise ValueError("resolution must be > 0.")

    origin_values = layout.get("origin", [0.0, 0.0])
    if not isinstance(origin_values, list) or len(origin_values) != 2:
        raise ValueError("origin must be [x, y].")
    origin_x = float(origin_values[0])
    origin_y = float(origin_values[1])

    world_width, world_height, width, height = _world_dimensions_from_layout(
        layout, resolution
    )
    robot_start = _parse_robot_start(layout, resolution, origin_x, origin_y)
    obstacles = _parse_obstacles(layout, resolution, origin_x, origin_y, width, height)

    _validate_world_entities(
        origin_x=origin_x,
        origin_y=origin_y,
        world_width=world_width,
        world_height=world_height,
        robot_start=robot_start,
        obstacles=obstacles,
    )

    occupied_cells = _rasterize_obstacles_to_cells(
        obstacles=obstacles,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        width=width,
        height=height,
    )

    return {
        "resolution": resolution,
        "origin": [origin_x, origin_y],
        "world_width": world_width,
        "world_height": world_height,
        "width": width,
        "height": height,
        "robot_start": robot_start,
        "obstacles": obstacles,
        "occupied_cells": occupied_cells,
    }


def _build_map_json(prepared: Dict) -> Dict:
    return {
        "width": int(prepared["width"]),
        "height": int(prepared["height"]),
        "resolution": float(prepared["resolution"]),
        "origin": [float(prepared["origin"][0]), float(prepared["origin"][1])],
        "occupied_cells": prepared["occupied_cells"],
    }


def _build_world_sdf(prepared: Dict) -> str:
    origin_x = float(prepared["origin"][0])
    origin_y = float(prepared["origin"][1])
    world_width = float(prepared["world_width"])
    world_height = float(prepared["world_height"])
    robot_start_x, robot_start_y = prepared["robot_start"]
    obstacles: List[Obstacle] = prepared["obstacles"]

    ground_center_x = origin_x + world_width / 2.0
    ground_center_y = origin_y + world_height / 2.0

    lines: List[str] = [
        '<?xml version="1.0" ?>',
        '<sdf version="1.6">',
        '  <world name="q_learning_world">',
        '    <physics name="1ms" type="ignored">',
        "      <max_step_size>0.001</max_step_size>",
        "      <real_time_factor>1.0</real_time_factor>",
        "    </physics>",
        '    <plugin filename="ignition-gazebo-physics-system" name="gz::sim::systems::Physics"/>',
        '    <plugin filename="ignition-gazebo-user-commands-system" name="gz::sim::systems::UserCommands"/>',
        '    <plugin filename="ignition-gazebo-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>',
        '    <light type="directional" name="sun">',
        "      <cast_shadows>true</cast_shadows>",
        "      <pose>0 0 20 0 0 0</pose>",
        "      <diffuse>1 1 1 1</diffuse>",
        "      <specular>0.3 0.3 0.3 1</specular>",
        "      <direction>-0.5 0.1 -0.9</direction>",
        "    </light>",
        '    <model name="ground_plane">',
        "      <static>true</static>",
        '      <link name="link">',
        f"        <pose>{ground_center_x:.3f} {ground_center_y:.3f} 0 0 0 0</pose>",
        '        <collision name="collision">',
        "          <geometry>",
        "            <box>",
        f"              <size>{world_width:.3f} {world_height:.3f} 0.05</size>",
        "            </box>",
        "          </geometry>",
        "        </collision>",
        '        <visual name="visual">',
        "          <geometry>",
        "            <box>",
        f"              <size>{world_width:.3f} {world_height:.3f} 0.05</size>",
        "            </box>",
        "          </geometry>",
        "          <material>",
        "            <ambient>0.8 0.8 0.8 1</ambient>",
        "            <diffuse>0.8 0.8 0.8 1</diffuse>",
        "          </material>",
        "        </visual>",
        "      </link>",
        "    </model>",
        '    <model name="vehicle_blue">',
        f"      <pose>{robot_start_x:.3f} {robot_start_y:.3f} 0.00 0 0 0</pose>",
        '      <link name="chassis">',
        "        <pose>0 0 0.10 0 0 0</pose>",
        "        <inertial><mass>2.0</mass><inertia><ixx>0.02</ixx><iyy>0.06</iyy><izz>0.07</izz></inertia></inertial>",
        '        <collision name="collision"><geometry><box><size>0.35 0.24 0.10</size></box></geometry></collision>',
        '        <visual name="visual"><geometry><box><size>0.35 0.24 0.10</size></box></geometry><material><ambient>0.3 0.5 1.0 1</ambient></material></visual>',
        "      </link>",
        '      <link name="left_wheel">',
        "        <pose>0 0.14 0.05 -1.5708 0 0</pose>",
        "        <inertial><mass>0.2</mass><inertia><ixx>0.0003</ixx><iyy>0.0003</iyy><izz>0.0003</izz></inertia></inertial>",
        '        <collision name="collision"><geometry><cylinder><radius>0.05</radius><length>0.03</length></cylinder></geometry></collision>',
        '        <visual name="visual"><geometry><cylinder><radius>0.05</radius><length>0.03</length></cylinder></geometry></visual>',
        "      </link>",
        '      <link name="right_wheel">',
        "        <pose>0 -0.14 0.05 -1.5708 0 0</pose>",
        "        <inertial><mass>0.2</mass><inertia><ixx>0.0003</ixx><iyy>0.0003</iyy><izz>0.0003</izz></inertia></inertial>",
        '        <collision name="collision"><geometry><cylinder><radius>0.05</radius><length>0.03</length></cylinder></geometry></collision>',
        '        <visual name="visual"><geometry><cylinder><radius>0.05</radius><length>0.03</length></cylinder></geometry></visual>',
        "      </link>",
        '      <link name="caster">',
        "        <pose>-0.14 0 0.03 0 0 0</pose>",
        "        <inertial><mass>0.1</mass><inertia><ixx>0.0001</ixx><iyy>0.0001</iyy><izz>0.0001</izz></inertia></inertial>",
        '        <collision name="collision"><geometry><sphere><radius>0.03</radius></sphere></geometry></collision>',
        '        <visual name="visual"><geometry><sphere><radius>0.03</radius></sphere></geometry></visual>',
        "      </link>",
        '      <joint name="left_wheel_joint" type="revolute"><parent>chassis</parent><child>left_wheel</child><axis><xyz>0 0 1</xyz></axis></joint>',
        '      <joint name="right_wheel_joint" type="revolute"><parent>chassis</parent><child>right_wheel</child><axis><xyz>0 0 1</xyz></axis></joint>',
        '      <joint name="caster_joint" type="ball"><parent>chassis</parent><child>caster</child></joint>',
        '      <plugin filename="ignition-gazebo-diff-drive-system" name="gz::sim::systems::DiffDrive">',
        "        <left_joint>left_wheel_joint</left_joint>",
        "        <right_joint>right_wheel_joint</right_joint>",
        "        <topic>/model/vehicle_blue/cmd_vel</topic>",
        "        <odom_topic>/model/vehicle_blue/odometry</odom_topic>",
        "        <frame_id>vehicle_blue/odom</frame_id>",
        "        <child_frame_id>vehicle_blue/chassis</child_frame_id>",
        "        <wheel_separation>0.28</wheel_separation>",
        "        <wheel_radius>0.05</wheel_radius>",
        "        <max_linear_velocity>0.7</max_linear_velocity>",
        "        <min_linear_velocity>-0.4</min_linear_velocity>",
        "        <max_angular_velocity>2.5</max_angular_velocity>",
        "        <min_angular_velocity>-2.5</min_angular_velocity>",
        "      </plugin>",
        "    </model>",
    ]

    for idx, obstacle in enumerate(obstacles):
        model_name = f"obs_{idx}_{_safe_name(obstacle.name)}"
        lines.extend(
            [
                f'    <model name="{model_name}">',
                "      <static>true</static>",
                f"      <pose>{obstacle.x:.3f} {obstacle.y:.3f} {0.5 * obstacle.height:.3f} 0 0 0</pose>",
                '      <link name="link">',
                '        <collision name="collision"><geometry><box>',
                f"          <size>{obstacle.size_x:.3f} {obstacle.size_y:.3f} {obstacle.height:.3f}</size>",
                "        </box></geometry></collision>",
                '        <visual name="visual"><geometry><box>',
                f"          <size>{obstacle.size_x:.3f} {obstacle.size_y:.3f} {obstacle.height:.3f}</size>",
                "        </box></geometry>",
                "        <material><ambient>0.7 0.2 0.2 1</ambient></material></visual>",
                "      </link>",
                "    </model>",
            ]
        )

    lines.extend(["  </world>", "</sdf>"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate matching occupancy-grid and Gazebo world assets."
    )
    parser.add_argument(
        "--layout-config",
        default=str(_resolve_default_layout()),
        help="Path to environment layout JSON.",
    )
    parser.add_argument(
        "--output-map",
        default="",
        help="Output occupancy-grid map JSON path (default: config/generated_grid_map.json).",
    )
    parser.add_argument(
        "--output-world",
        default="",
        help="Output Gazebo world SDF path (default: worlds/q_learning_world.sdf).",
    )
    args = parser.parse_args()

    layout_path = Path(args.layout_config).expanduser().resolve()
    if not layout_path.exists():
        raise FileNotFoundError(f"Layout file not found: {layout_path}")

    with layout_path.open("r", encoding="utf-8") as file:
        layout = json.load(file)

    prepared = _prepare_layout(layout)

    output_map = (
        Path(args.output_map).expanduser().resolve()
        if args.output_map
        else layout_path.parent / "generated_grid_map.json"
    )
    output_world = (
        Path(args.output_world).expanduser().resolve()
        if args.output_world
        else layout_path.parent.parent / "worlds" / "q_learning_world.sdf"
    )

    output_map.parent.mkdir(parents=True, exist_ok=True)
    output_world.parent.mkdir(parents=True, exist_ok=True)

    map_data = _build_map_json(prepared)
    with output_map.open("w", encoding="utf-8") as file:
        json.dump(map_data, file, indent=2)
        file.write("\n")

    world_sdf = _build_world_sdf(prepared)
    with output_world.open("w", encoding="utf-8") as file:
        file.write(world_sdf)

    print(f"Layout file:      {layout_path}")
    print(f"World size (m):   {prepared['world_width']:.3f} x {prepared['world_height']:.3f}")
    print(f"Grid size:        {prepared['width']} x {prepared['height']} cells")
    print(f"Obstacle count:   {len(prepared['obstacles'])}")
    print(f"Generated map:    {output_map}")
    print(f"Generated world:  {output_world}")


if __name__ == "__main__":
    main()
