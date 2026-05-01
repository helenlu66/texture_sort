import glob

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class GelSightNode(Node):
    def __init__(self):
        super().__init__('gelsight_node')

        self.declare_parameter('device_index', 0)
        self.declare_parameter('width', 320)
        self.declare_parameter('height', 240)
        self.declare_parameter('border_fraction', 0.15)
        self.declare_parameter('image_topic', '/gelsight/image_raw')
        self.declare_parameter('frame_id', 'gelsight_frame')
        self.declare_parameter('fps', 25)

        device_index    = int(self.get_parameter('device_index').value)
        self._width     = int(self.get_parameter('width').value)
        self._height    = int(self.get_parameter('height').value)
        self._border    = float(self.get_parameter('border_fraction').value)
        image_topic     = str(self.get_parameter('image_topic').value)
        self._frame_id  = str(self.get_parameter('frame_id').value)
        fps             = int(self.get_parameter('fps').value)

        device = self._resolve_device(device_index)
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise RuntimeError(f'Could not open GelSight device: {device}')

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        self._bridge = CvBridge()
        self._pub = self.create_publisher(Image, image_topic, qos_profile_sensor_data)
        self.create_timer(1.0 / fps, self._publish_frame)

        self.get_logger().info(f'gelsight_node ready — device: {device}, topic: {image_topic}')

    def _resolve_device(self, index: int):
        paths = sorted(glob.glob('/dev/v4l/by-id/*'))
        if paths and index < len(paths):
            return paths[index]
        return index

    def _publish_frame(self):
        ret, frame = self._cap.read()
        if not ret:
            self.get_logger().warn('Failed to read frame', throttle_duration_sec=2.0)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._border > 0.0:
            h, w = frame_rgb.shape[:2]
            bh, bw = int(h * self._border), int(w * self._border)
            frame_rgb = frame_rgb[bh:h - bh, bw:w - bw]

        frame_rgb = cv2.resize(frame_rgb, (self._width, self._height))

        msg = self._bridge.cv2_to_imgmsg(frame_rgb, encoding='rgb8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        self._pub.publish(msg)

    def destroy_node(self):
        self._cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GelSightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
