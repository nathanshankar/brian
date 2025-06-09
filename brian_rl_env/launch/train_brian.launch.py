# train_brian.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_brian_pybullet = get_package_share_directory('brian_pybullet')
    pkg_brian_rl_env = get_package_share_directory('brian_rl_env')

    # Path to your URDF
    urdf_path = os.path.join(get_package_share_directory('brian_description'), 'urdf', 'brian.urdf')

    return LaunchDescription([
        Node(
            package='brian_pybullet',
            executable='brian_pybullet_node.py',
            name='brian_pybullet_sim',
            output='screen',
            parameters=[
                {'robot_name': 'brian'},
                {'urdf_path': urdf_path},
            ]
        ),
        Node(
            package='brian_rl_env',
            executable='train_brian',
            name='brian_rl_trainer',
            output='screen',
            # Set python buffering to 1 for real-time log output
            # (optional, but useful for debugging)
            # You might need to adjust the path to your python executable
            # and ensure train_brian.py is executable.
            # For simplicity, if train_brian.py is in scripts/, ensure your setup.py marks it.
            # If `executable` doesn't find it, ensure it's in a path that `ros2 run` can find.
            # Best practice is to register it as an entry_point in setup.py
        )
    ])