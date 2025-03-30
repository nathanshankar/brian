from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import Command, PathJoinSubstitution
from launch.substitutions import FindExecutable

def generate_launch_description():

    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        PathJoinSubstitution([
            FindPackageShare('brian_description'),
            'urdf',
            'brian.urdf'
        ])
    ])

    # Robot State Publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_description_content}, {'rate': 100}],
        remappings=[('joint_states', '/brian/joints_state')]
    )

    # Joint state publisher node
    joint_state_publisher_gui = Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            parameters=[robot_description_content],
            remappings=[('joint_states', '/brian/joints_state')]
        )

    # Simulated joints state node
    pybullet_node = Node(
        package='brian_pybullet',
        executable='brian_pybullet_node.py',
        name='brian_pybullet',
        output='screen'
    )

    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', os.path.join(
            get_package_share_directory('brian_description'),
            'rviz',
            'show_model.rviz'
        )]
    )

    return LaunchDescription([
        robot_state_publisher_node,
        # joint_state_publisher_gui,
        pybullet_node,
        rviz_node
    ])