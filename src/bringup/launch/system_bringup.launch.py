from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(package='perception', executable='vision_node', name='vision_node', output='screen'),
        Node(package='perception', executable='tactile_node', name='tactile_node', output='screen'),
        Node(package='manipulation', executable='manipulation_node', name='manipulation_node', output='screen'),
        Node(package='navigation', executable='delivery_node', name='turtlebot_delivery_node', output='screen'),
        Node(package='task_manager', executable='task_manager_node', name='task_manager_node', output='screen'),
    ])
