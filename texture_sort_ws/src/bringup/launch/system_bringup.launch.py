from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(package='texture_sorting_perception', executable='vision_node', name='vision_node', output='screen'),
        Node(package='texture_sorting_perception', executable='tactile_node', name='tactile_node', output='screen'),
        Node(package='texture_sorting_manipulation', executable='manipulation_node', name='manipulation_node', output='screen'),
        Node(package='texture_sorting_navigation', executable='delivery_node', name='turtlebot_delivery_node', output='screen'),
        Node(package='texture_sorting_task_manager', executable='task_manager_node', name='task_manager_node', output='screen'),
    ])
