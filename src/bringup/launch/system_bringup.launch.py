from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description() -> LaunchDescription:
    params_file = os.path.join(
        get_package_share_directory('bringup'),
        'config',
        'system_params.yaml',
    )

    return LaunchDescription([
        Node(package='realsense2_camera', executable='realsense2_camera_node', name='external_depth_camera', output='screen', parameters=[params_file]),
        Node(package='perception', executable='external_cam_node', name='external_cam_node', output='screen', parameters=[params_file]),
        Node(package='perception', executable='kinova_rtsp_bridge', name='kinova_rtsp_bridge', output='screen', parameters=[params_file]),
        Node(package='realsense2_camera', executable='realsense2_camera_node', name='wrist_depth_camera', output='screen', parameters=[params_file]),
        Node(package='perception', executable='kinova_wrist_cam_node', name='kinova_wrist_cam_node', output='screen', parameters=[params_file]),
        Node(package='rqt_image_view', executable='rqt_image_view', name='kinova_wrist_cam_view', output='screen', arguments=['/wrist_camera/color/detected_bboxes_image']),
        Node(package='perception', executable='tactile_node', name='tactile_node', output='screen'),
        Node(package='manipulation', executable='manipulation_node', name='manipulation_node', output='screen'),
        Node(package='navigation', executable='delivery_node', name='turtlebot_delivery_node', output='screen'),
        Node(package='task_manager', executable='task_manager_node', name='task_manager_node', output='screen'),
    ])
