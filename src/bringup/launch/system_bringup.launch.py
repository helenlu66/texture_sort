from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import os
import yaml



def generate_launch_description() -> LaunchDescription:
    params_file = os.path.join(
        get_package_share_directory('bringup'),
        'config',
        'system_params.yaml',
    )

    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    rs = params.get('external_depth_camera', {}).get('ros__parameters', {})
    kv = params.get('kinova_vision', {}).get('ros__parameters', {})

    rs_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py',
            )
        ),
        launch_arguments=rs.items(),
    )

    kinova_vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('kinova_vision'),
                'launch',
                'kinova_vision.launch.py',
            )
        ),
        launch_arguments=kv.items(),
    )

    return LaunchDescription([
        rs_launch,
        kinova_vision_launch,
        Node(
            package='apriltag_ros',
            executable='apriltag_node',
            name='apriltag',
            output='screen',
            parameters=[params_file],
            remappings=[
                ('image_rect', '/external_camera/color/image_raw'),
                ('camera_info', '/external_camera/color/camera_info'),
                ('/detections', '/apriltag/detections'),
            ],
        ),
        Node(package='perception', executable='apriltag_overlay', name='apriltag_overlay', output='screen', parameters=[params_file]),
        Node(package='rqt_image_view', executable='rqt_image_view', name='external_camera_view', output='screen', arguments=['/detections_image']),
        Node(package='rqt_image_view', executable='rqt_image_view', name='wrist_camera_view', output='screen', arguments=['/wrist_camera/color/image_raw']),
        # Node(package='perception', executable='external_cam_node', name='external_cam_node', output='screen', parameters=[params_file]),
        # Node(package='perception', executable='tactile_node', name='tactile_node', output='screen'),
        # Node(package='manipulation', executable='manipulation_node', name='manipulation_node', output='screen'),
        # Node(package='navigation', executable='delivery_node', name='turtlebot_delivery_node', output='screen'),
        # Node(package='task_manager', executable='task_manager_node', name='task_manager_node', output='screen'),
    ])
