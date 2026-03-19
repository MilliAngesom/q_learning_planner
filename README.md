# q_learning_planner

ROS 2 Humble package for grid-based Q-learning path planning with known occupancy maps.

## 0) Workspace setup (for any user)

Set your workspace path and package source path once per terminal:
Open your terminal and navigate to your ROS2 workspace using `cd`.

```bash
export WS=$(pwd)
export PKG_SRC=$WS/src/q_learning_planner
cd $WS
```

If you cloned this repository into a different folder name, update `PKG_SRC` accordingly.

## 1) Build

```bash
cd $WS
source /opt/ros/humble/setup.bash
colcon build --packages-select q_learning_planner
source install/setup.bash
```

## Run by scenario

Run these from `$WS` after sourcing:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

### 1) Train planner, then run full system

```bash
ros2 launch q_learning_planner full_system.launch.py \
  train_rl:=true \
  model_path:=/tmp/q_learning_q_tables.npz \
  rviz:=true
```

### 2) Run full system without training

Use this only if the Q-table file already exists.

```bash
ros2 launch q_learning_planner full_system.launch.py \
  train_rl:=false \
  model_path:=/tmp/q_learning_q_tables.npz \
  rviz:=true
```

### 3) You changed the environment

1. Edit:
   - `$PKG_SRC/config/environment_layout.json`
2. Regenerate map + world from the layout:

```bash
ros2 run q_learning_planner generate_env_assets \
  --layout-config $PKG_SRC/config/environment_layout.json \
  --output-map $PKG_SRC/config/generated_grid_map.json \
  --output-world $PKG_SRC/worlds/q_learning_world.sdf
```

3. Rebuild so launch defaults in `install/` get updated:

```bash
colcon build --packages-select q_learning_planner
source install/setup.bash
```

4. Recommended: retrain for the new map, then run:

```bash
ros2 launch q_learning_planner full_system.launch.py \
  train_rl:=true \
  world:=$PKG_SRC/worlds/q_learning_world.sdf \
  map_config_path:=$PKG_SRC/config/generated_grid_map.json \
  model_path:=/tmp/q_learning_q_tables.npz
```

If you skip retraining after environment changes, planning quality may degrade because the Q-table was learned on a different map.

## 2) Define one source of truth for environment layout

Edit:

- `$PKG_SRC/config/environment_layout.json`

This file defines:

- physical world size (`world_size`) in meters
- `resolution` and `origin`
- robot start (`robot_start`) in world coordinates
- obstacle list (`obstacles`) with explicit world position and size (meters)

## 3) Generate matching assets (map + Gazebo world)

```bash
cd $WS
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run q_learning_planner generate_env_assets \
  --layout-config $PKG_SRC/config/environment_layout.json \
  --output-map $PKG_SRC/config/generated_grid_map.json \
  --output-world $PKG_SRC/worlds/q_learning_world.sdf
```

Now both files are synchronized:

- `config/generated_grid_map.json` (for Q-learning training/planning)
- `worlds/q_learning_world.sdf` (for Gazebo simulation)

## 4) Train Q-tables on generated map

```bash
cd $WS
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch q_learning_planner train_q_table.launch.py \
  map_config_path:=$PKG_SRC/config/generated_grid_map.json \
  output_q_table_path:=/tmp/q_learning_q_tables.npz
```

This now also saves per-episode training history to a companion file by default:

- `/tmp/q_learning_q_tables.training_history.npz`

You can override that path with `output_training_history_path:=/your/path/training_history.npz`.

## 4b) Plot full training curves from an actual run

```bash
cd $WS
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run q_learning_planner plot_training_history \
  /tmp/q_learning_q_tables.training_history.npz \
  --window 200 \
  --goal-boundaries
```

This opens an interactive Matplotlib window with full per-episode reward, steps, success-rate, and epsilon curves from the saved run history.

If you also want a PNG:

```bash
ros2 run q_learning_planner plot_training_history \
  /tmp/q_learning_q_tables.training_history.npz \
  --output /tmp/q_learning_training_curves.png \
  --no-show
```

## 5) Run Gazebo + planner + follower + RViz

```bash
cd $WS
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch q_learning_planner sim_with_planner.launch.py \
  model_path:=/tmp/q_learning_q_tables.npz
```

## 5b) One launch for train + run (skip training optional)

Train then run:

```bash
ros2 launch q_learning_planner full_system.launch.py \
  train_rl:=true \
  model_path:=/tmp/q_learning_q_tables.npz
```

Skip training and run directly:

```bash
ros2 launch q_learning_planner full_system.launch.py \
  train_rl:=false \
  model_path:=/tmp/q_learning_q_tables.npz
```

For safer RL paths (farther from obstacles), retrain with stronger clearance penalty:

```bash
ros2 launch q_learning_planner train_q_table.launch.py \
  map_config_path:=$PKG_SRC/config/generated_grid_map.json \
  output_q_table_path:=/tmp/q_learning_q_tables.npz \
  desired_clearance_cells:=2 \
  clearance_penalty_weight:=2.0
```

`sim_with_planner.launch.py` now auto-starts:

- Gazebo world
- map publisher (`/map`) using `generated_grid_map.json`
- path planner
- path follower
- RViz with preloaded config

Path tracking uses the heading-based local controller (`q_path_follower`).

So when RViz opens you already see:

- occupied/free grid map (`/map`)
- planned path (`/q_learning/path`)
- clicked goal pose (`/goal_pose`)
- robot body marker (`/q_learning/robot_body`)

## 6) Set goal from RViz (recommended)

In RViz:

1. Select the `2D Goal Pose` tool in the top toolbar.
2. Click on the map to set goal position and drag to set heading.
3. The goal is published on `/goal_pose` and the planner replans immediately.
4. If your click lands on an occupied cell, planner rejects it and prints an error.

If you don't want RViz:

```bash
ros2 launch q_learning_planner sim_with_planner.launch.py \
  model_path:=/tmp/q_learning_q_tables.npz \
  rviz:=false
```

Optional controller tuning for tighter collision avoidance:

```bash
ros2 launch q_learning_planner sim_with_planner.launch.py \
  model_path:=/tmp/q_learning_q_tables.npz \
  simplify_path:=false \
  waypoint_tolerance:=0.20 \
  robot_radius:=0.24 \
  obstacle_clearance:=0.12 \
  linear_gain:=0.8 \
  angular_gain:=2.8
```

## 7) Optional: send goal from terminal

Goal cells must be converted to world coordinates:

- `x_world = origin_x + (cell_x + 0.5) * resolution`
- `y_world = origin_y + (cell_y + 0.5) * resolution`

Example for `goal_cell=(13, 13)` with `origin=(0,0)` and `resolution=0.5`:
- `x_world=6.75`, `y_world=6.75`

```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
"{header: {frame_id: 'map'}, pose: {position: {x: 6.75, y: 6.75, z: 0.0}, orientation: {w: 1.0}}}"
```

Inspect:

```bash
ros2 topic echo /q_learning/path --once
ros2 topic echo /model/vehicle_blue/cmd_vel
```

## Environment matching checklist

1. Keep `resolution` identical between layout/map/world.
2. Keep `origin` identical between layout/map/world.
3. Keep `world_size` fixed if you want resolution changes to affect only grid-cell count.
4. Define obstacles in `obstacles` (world coordinates/sizes), not by cell index.
5. Regenerate assets every time layout changes (`resolution`, `world_size`, `obstacles`, start pose).
6. Train on `generated_grid_map.json`, not an old map file.
7. Use Gazebo ground-truth start (`use_ground_truth_start:=true`) in simulation (default in `sim_with_planner.launch.py`).

## Runtime topics

- Subscribes to `/goal_pose` (`geometry_msgs/PoseStamped`)
- Planner start from Gazebo ground-truth (`/world/q_learning_world/dynamic_pose/info`) by default in `sim_with_planner.launch.py`
- Publishes `/q_learning/path` (`nav_msgs/Path`)
- Follower publishes `/model/vehicle_blue/cmd_vel` (`geometry_msgs/Twist`)
- Follower publishes `/q_learning/robot_pose` (`geometry_msgs/PoseStamped`, frame `map`)
- Follower publishes `/q_learning/robot_body` (`visualization_msgs/Marker`)

If behavior looks inconsistent across runs (wrong start cell, odd odometry), stop stale processes first:

```bash
pkill -f "ign gazebo" || true
pkill -f "ros2 launch q_learning_planner" || true
pkill -f "parameter_bridge" || true
```

If Gazebo robot still does not move:

1. Make sure simulation is running (play button in Gazebo is not paused).
2. Confirm commands exist: `ros2 topic echo /model/vehicle_blue/cmd_vel`.
3. Confirm odometry changes: `ros2 topic echo /model/vehicle_blue/odometry`.
