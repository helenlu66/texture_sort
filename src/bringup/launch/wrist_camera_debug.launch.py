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

    wrist_rgb_params = {
        'image_topic': '/wrist_camera/color/image_raw',
        'camera_info_topic': '/wrist_camera/color/camera_info',
    }

    vnn_params = {
        'image_topic': '/wrist_camera/color/image_raw',
    }

    return LaunchDescription([
        Node(
            package='perception',
            executable='kinova_rtsp_bridge',
            name='kinova_rtsp_bridge',
            output='screen',
            parameters=[params_file, wrist_rgb_params],
        ),
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='wrist_rgb_view',
            output='screen',
            arguments=['/wrist_camera/color/image_raw'],
        ),
        Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='wrist_detections_view',
            output='screen',
            arguments=['/wrist_camera/color/vnn_detections'],
        ),
    ])
