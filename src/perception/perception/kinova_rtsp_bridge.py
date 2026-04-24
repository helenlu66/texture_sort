import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class KinovaRtspBridge(Node):
    def __init__(self) -> None:
        super().__init__("kinova_rtsp_bridge")

        self.declare_parameter("rtsp_url", "rtsp://192.168.1.10/color")
        self.declare_parameter("image_topic", "/wrist_camera/image_raw")
        self.declare_parameter("camera_info_topic", "/wrist_camera/camera_info")
        self.declare_parameter("frame_id", "wrist_camera_color_optical_frame")
        self.declare_parameter("fps", 15.0)
        self.declare_parameter("reconnect_sec", 1.0)
        self.declare_parameter("fx", 0.0)
        self.declare_parameter("fy", 0.0)
        self.declare_parameter("cx", 0.0)
        self.declare_parameter("cy", 0.0)
        self.declare_parameter("distortion_model", "plumb_bob")
        self.declare_parameter("distortion_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])

        self.rtsp_url = self.get_parameter("rtsp_url").get_parameter_value().string_value
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.fps = self.get_parameter("fps").get_parameter_value().double_value
        self.reconnect_sec = self.get_parameter("reconnect_sec").get_parameter_value().double_value
        self.fx = self.get_parameter("fx").get_parameter_value().double_value
        self.fy = self.get_parameter("fy").get_parameter_value().double_value
        self.cx = self.get_parameter("cx").get_parameter_value().double_value
        self.cy = self.get_parameter("cy").get_parameter_value().double_value
        self.distortion_model = (
            self.get_parameter("distortion_model").get_parameter_value().string_value
        )
        self.distortion_coeffs = list(
            self.get_parameter("distortion_coeffs").get_parameter_value().double_array_value
        )

        self.image_pub = self.create_publisher(Image, self.image_topic, 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, self.camera_info_topic, 10)

        self.cap = None
        self.last_connect_attempt = 0.0
        self.camera_info_msg = None

        period = 1.0 / max(self.fps, 1.0)
        self.timer = self.create_timer(period, self._tick)
        self.get_logger().info(f"RTSP bridge configured for {self.rtsp_url}")

    def _connect_if_needed(self) -> bool:
        if self.cap is not None and self.cap.isOpened():
            return True

        now = time.time()
        if now - self.last_connect_attempt < self.reconnect_sec:
            return False

        self.last_connect_attempt = now
        if self.cap is not None:
            self.cap.release()

        self.get_logger().info(f"Connecting to RTSP stream: {self.rtsp_url}")
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            self.get_logger().warn("Failed to open RTSP stream; retrying...")
            return False

        self.get_logger().info("RTSP stream connected")
        return True

    def _build_camera_info(self, width: int, height: int) -> CameraInfo:
        msg = CameraInfo()
        msg.width = width
        msg.height = height
        msg.distortion_model = self.distortion_model

        d = self.distortion_coeffs[:]
        if len(d) < 5:
            d.extend([0.0] * (5 - len(d)))
        msg.d = d

        fx = self.fx if self.fx > 0.0 else float(width)
        fy = self.fy if self.fy > 0.0 else float(height)
        cx = self.cx if self.cx > 0.0 else float(width) / 2.0
        cy = self.cy if self.cy > 0.0 else float(height) / 2.0

        msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    def _tick(self) -> None:
        if not self._connect_if_needed():
            return

        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            self.get_logger().warn("RTSP read failed; reconnecting...")
            self.cap.release()
            self.cap = None
            return

        # ROS standard for Image is RGB8. OpenCV yields BGR.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, _ = frame_rgb.shape

        img = Image()
        img.header.stamp = self.get_clock().now().to_msg()
        img.header.frame_id = self.frame_id
        img.height = height
        img.width = width
        img.encoding = "rgb8"
        img.is_bigendian = False
        img.step = width * 3
        img.data = np.asarray(frame_rgb, dtype=np.uint8).tobytes()

        if self.camera_info_msg is None or (
            self.camera_info_msg.width != width or self.camera_info_msg.height != height
        ):
            self.camera_info_msg = self._build_camera_info(width, height)

        cam_info = self.camera_info_msg
        cam_info.header.stamp = img.header.stamp
        cam_info.header.frame_id = self.frame_id

        self.image_pub.publish(img)
        self.camera_info_pub.publish(cam_info)

    def destroy_node(self) -> bool:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = KinovaRtspBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()