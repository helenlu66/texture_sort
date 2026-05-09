import copy

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image


class WristCameraQoSRelay(Node):
    def __init__(self) -> None:
        super().__init__('wrist_camera_qos_relay')

        self.declare_parameter('input_image_topic', '/wrist_camera/color/image_raw')
        self.declare_parameter('input_camera_info_topic', '/wrist_camera/color/camera_info')
        self.declare_parameter('output_image_topic', '/wrist_camera/color/image_raw_best_effort')
        self.declare_parameter('output_camera_info_topic', '/wrist_camera/color/camera_info_best_effort')

        input_image_topic = str(self.get_parameter('input_image_topic').value)
        input_camera_info_topic = str(self.get_parameter('input_camera_info_topic').value)
        output_image_topic = str(self.get_parameter('output_image_topic').value)
        output_camera_info_topic = str(self.get_parameter('output_camera_info_topic').value)

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self._image_pub = self.create_publisher(Image, output_image_topic, sensor_qos)
        self._camera_info_pub = self.create_publisher(CameraInfo, output_camera_info_topic, sensor_qos)
        self._latest_camera_info = None
        self.create_subscription(Image, input_image_topic, self._image_cb, reliable_qos)
        self.create_subscription(CameraInfo, input_camera_info_topic, self._camera_info_cb, reliable_qos)

        self.get_logger().info(
            f'Relaying {input_image_topic} -> {output_image_topic} and '
            f'{input_camera_info_topic} -> {output_camera_info_topic}'
        )

    def _image_cb(self, msg: Image) -> None:
        self._image_pub.publish(msg)
        if self._latest_camera_info is None:
            return
        camera_info = copy.deepcopy(self._latest_camera_info)
        camera_info.header = msg.header
        self._camera_info_pub.publish(camera_info)

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self._latest_camera_info = msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WristCameraQoSRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
