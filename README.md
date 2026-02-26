# q_learning_planner

ROS 2 Humble package for grid-based Q-learning path planning with known occupancy maps.

## Quick start

1. Build package:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select q_learning_planner
source install/setup.bash
```

2. Train Q-tables on sample map:

```bash
ros2 launch q_learning_planner train_q_table.launch.py
```

3. Plan and publish path once:

```bash
ros2 launch q_learning_planner path_planner.launch.py \
  model_path:=/tmp/q_learning_q_tables.npz \
  plan_on_startup:=true \
  start_cell_x:=1 start_cell_y:=1 \
  goal_cell_x:=13 goal_cell_y:=13
```

4. Inspect path:

```bash
ros2 topic echo /q_learning/path --once
```

## Runtime topics

- Subscribes to `/goal_pose` (`geometry_msgs/PoseStamped`)
- Optional start from `/initialpose` (`geometry_msgs/PoseWithCovarianceStamped`)
- Optional start from odometry topic (default `/model/vehicle_blue/odometry`)
- Publishes `/q_learning/path` (`nav_msgs/Path`)
