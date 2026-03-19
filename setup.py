import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'q_learning_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.json')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools', 'numpy', 'matplotlib'],
    zip_safe=True,
    maintainer='milli',
    maintainer_email='million.asefaw@uqtr.ca',
    description='Q-learning based path and motion planner nodes for ROS 2.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'train_q_table = q_learning_planner.train_q_table_node:main',
            'plot_training_history = q_learning_planner.plot_training_history:main',
            'q_path_planner = q_learning_planner.path_planner_node:main',
            'q_path_follower = q_learning_planner.path_follower_node:main',
            'q_map_publisher = q_learning_planner.map_publisher_node:main',
            'generate_env_assets = q_learning_planner.generate_env_assets:main',
        ],
    },
)
