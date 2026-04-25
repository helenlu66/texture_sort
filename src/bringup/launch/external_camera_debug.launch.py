from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os


def generate_launch_description() -> LaunchDescription:
    rs_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py',
            )
        ),
        launch_arguments={
            'camera_name': 'external_camera',
            'camera_namespace': '',
            'serial_no': "'925622071555'",
            'enable_color': 'true',
            'enable_depth': 'true',
            'pointcloud.enable': 'false',
        }.items(),
    )

    return LaunchDescription([
        rs_launch,
    ])
