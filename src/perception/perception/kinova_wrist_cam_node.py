#!/usr/bin/env python3
"""Kinova wrist RGB-D localization node for ROS2 Jazzy.

This node synchronizes compressed RGB and depth images, performs OWLv2
open-vocabulary detection, projects detections to 3D using depth, transforms the
points to base_link using Kinova EE feedback, and publishes debug outputs.
"""

import math
from typing import Any, List, Optional, Tuple

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from visualization_msgs.msg import Marker, MarkerArray


def euler_xyz_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Return quaternion (x, y, z, w) from XYZ intrinsic euler angles."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Return 3x3 rotation matrix from quaternion (x, y, z, w)."""
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


class KinovaWristCamNode(Node):
    """ROS2 version of the Noetic bbox+depth localizer."""

    def __init__(self) -> None:
        super().__init__("kinova_wrist_cam_node")
        self.bridge = CvBridge()

        self.declare_parameter("robot_name", "my_gen3")
        self.declare_parameter("process_rate", 0.5)
        self.declare_parameter("color_info_topic", "/camera/color/camera_info")
        self.declare_parameter("depth_info_topic", "/camera/depth/camera_info")
        self.declare_parameter("depth_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter("rgb_topic", "/camera/color/image_rect_color/compressed")
        self.declare_parameter("detector_labels", ["pan", "tomatoe", "eggplant", "red basket", "leek"])
        self.declare_parameter("detection_threshold", 0.15)
        self.declare_parameter("ee_to_cam_translation", [0.0, 0.062, -0.114])
        self.declare_parameter("ee_to_cam_quaternion", [0.0, 0.0, -0.5, 0.0])
        self.declare_parameter("depth_to_color_translation", [-0.023, -0.005, 0.0])
        self.declare_parameter("enable_owlv2", True)

        self.robot_name = str(self.get_parameter("robot_name").value)
        self.process_rate = float(self.get_parameter("process_rate").value)
        self.color_info_topic = str(self.get_parameter("color_info_topic").value)
        self.depth_info_topic = str(self.get_parameter("depth_info_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        labels = self.get_parameter("detector_labels").value
        self.text_labels = [list(labels)]
        self.detection_threshold = float(self.get_parameter("detection_threshold").value)
        self.ee_t = np.array(self.get_parameter("ee_to_cam_translation").value, dtype=np.float64)
        self.ee_q = np.array(self.get_parameter("ee_to_cam_quaternion").value, dtype=np.float64)
        self.t_depth_to_color = np.array(self.get_parameter("depth_to_color_translation").value, dtype=np.float64)
        self.enable_owlv2 = bool(self.get_parameter("enable_owlv2").value)

        self._last_process_time = self.get_clock().now()
        self._last_warn_ns = {}

        self.fx_c = self.fy_c = self.cx_c = self.cy_c = None
        # self.fx_d = self.fy_d = self.cx_d = self.cy_d = None
        self.K_color = None
        self.D_color = None
        # self.K_depth = None
        # self.D_depth = None
        self.color_W = self.color_H = None
        # self.depth_W = self.depth_H = None
        self.color_info_ready = False
        # self.depth_info_ready = False

        self.current_ee_pose: Optional[Tuple[float, float, float, float, float, float, float]] = None
        self.base_feedback_type = None

        self.image_pub = self.create_publisher(Image, "/camera/color/detected_bboxes_image", 10)
        self.depth_pub = self.create_publisher(Image, "/camera/depth/detected_bboxes_image", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/detected_objects_marker", 10)
        self.object_pub = self.create_publisher(PointStamped, "/detected_object_point", 10)

        self._color_info_sub = self.create_subscription(
            CameraInfo, self.color_info_topic, self.color_info_callback, 10
        )
        # self._depth_info_sub = self.create_subscription(
        #     CameraInfo, self.depth_info_topic, self.depth_info_callback, 10
        # )
        try:
            import importlib

            kortex_msg = importlib.import_module("kortex_driver.msg")
            self.base_feedback_type = getattr(kortex_msg, "BaseCyclic_Feedback", None)
        except Exception:
            self.base_feedback_type = None

        if self.base_feedback_type is not None:
            self.create_subscription(
                self.base_feedback_type,
                f"/{self.robot_name}/base_feedback",
                self.feedback_callback,
                10,
            )
        else:
            self.get_logger().warn(
                "kortex_driver is unavailable; EE feedback subscription disabled."
            )

        # self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)
        # self.color_sub = message_filters.Subscriber(self, CompressedImage, self.rgb_topic)
        # self.ts = message_filters.ApproximateTimeSynchronizer(
        #     [self.depth_sub, self.color_sub], queue_size=100, slop=0.05
        # )
        # self.ts.registerCallback(self.sync_callback)
        self.create_subscription(CompressedImage, self.rgb_topic, self.rgb_callback, 10)

        self.processor = None
        self.model = None
        self.torch = None
        self.device = "cpu"
        self._init_detector_if_enabled()

        self.get_logger().info("kinova_wrist_cam_node ready - waiting for synced frames")

    def _init_detector_if_enabled(self) -> None:
        if not self.enable_owlv2:
            self.get_logger().warn("OWLv2 disabled via parameter; node will only publish debug streams.")
            return

        try:
            import importlib

            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            Owlv2Processor = getattr(transformers, "Owlv2Processor")
            Owlv2ForObjectDetection = getattr(transformers, "Owlv2ForObjectDetection")
        except Exception as exc:
            self.get_logger().warn(
                f"OWLv2 dependencies unavailable ({exc}); detection disabled."
            )
            self.enable_owlv2 = False
            return

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = (
            Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
            .to(self.device)
            .eval()
        )
        self.get_logger().info(f"OWLv2 loaded on {self.device}")

    def _warn_throttle(self, key: str, period_s: float, text: str) -> None:
        now_ns = self.get_clock().now().nanoseconds
        prev_ns = self._last_warn_ns.get(key, 0)
        if now_ns - prev_ns >= int(period_s * 1e9):
            self._last_warn_ns[key] = now_ns
            self.get_logger().warn(text)

    @property
    def intrinsics_ready(self) -> bool:
        return self.color_info_ready  # and self.depth_info_ready

    def color_info_callback(self, msg: CameraInfo) -> None:
        if self.color_info_ready:
            return

        k = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.fx_c, self.fy_c = k[0, 0], k[1, 1]
        self.cx_c, self.cy_c = k[0, 2], k[1, 2]
        self.K_color = k
        self.D_color = np.array(msg.d, dtype=np.float64)
        self.color_W, self.color_H = msg.width, msg.height
        self.color_info_ready = True
        if self._color_info_sub is not None:
            self.destroy_subscription(self._color_info_sub)
            self._color_info_sub = None
        self.get_logger().info(
            f"Color intrinsics ready fx={self.fx_c:.2f} fy={self.fy_c:.2f} cx={self.cx_c:.2f} cy={self.cy_c:.2f}"
        )

    # def depth_info_callback(self, msg: CameraInfo) -> None:
    #     if self.depth_info_ready:
    #         return
    #     k = np.array(msg.k, dtype=np.float64).reshape(3, 3)
    #     self.fx_d, self.fy_d = k[0, 0], k[1, 1]
    #     self.cx_d, self.cy_d = k[0, 2], k[1, 2]
    #     self.K_depth = k
    #     self.D_depth = np.array(msg.d, dtype=np.float64)
    #     self.depth_W, self.depth_H = msg.width, msg.height
    #     self.depth_info_ready = True
    #     if self._depth_info_sub is not None:
    #         self.destroy_subscription(self._depth_info_sub)
    #         self._depth_info_sub = None

    def feedback_callback(self, msg: Any) -> None:
        try:
            tx = msg.base.tool_pose_x
            ty = msg.base.tool_pose_y
            tz = msg.base.tool_pose_z
            rx = math.radians(msg.base.tool_pose_theta_x)
            ry = math.radians(msg.base.tool_pose_theta_y)
            rz = math.radians(msg.base.tool_pose_theta_z)
            qx, qy, qz, qw = euler_xyz_to_quaternion(rx, ry, rz)
            self.current_ee_pose = (tx, ty, tz, qx, qy, qz, qw)
        except Exception as exc:
            self._warn_throttle("ee_pose", 5.0, f"Failed reading EE pose: {exc}")

    def sync_callback(self, depth_msg: Image, color_msg: CompressedImage) -> None:
        now = self.get_clock().now()
        dt = (now.nanoseconds - self._last_process_time.nanoseconds) * 1e-9
        if self.process_rate > 0.0 and dt < 1.0 / self.process_rate:
            return
        self._last_process_time = now

        if not self.intrinsics_ready:
            self._warn_throttle("intrinsics", 2.0, "Intrinsics not ready, skipping frame")
            return

        try:
            np_arr = np.frombuffer(color_msg.data, np.uint8)
            bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr_img is None:
                raise ValueError("imdecode returned None")
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            self._warn_throttle("rgb_decode", 5.0, f"Color decode failed: {exc}")
            return

        try:
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            self._warn_throttle("depth_decode", 5.0, f"Depth decode failed: {exc}")
            return

        try:
            depth_registered = self.register_depth(depth_raw)
        except Exception as exc:
            self._warn_throttle("depth_register", 5.0, f"Depth registration failed: {exc}")
            return

        self.process_frame(rgb_img, depth_registered)

    def register_depth(self, depth_img: np.ndarray) -> np.ndarray:
        d = depth_img.astype(np.float32)
        if d.size == 0:
            raise ValueError("empty depth image")
        if float(np.nanmax(d)) > 100.0:
            d /= 1000.0

        h_c, w_c = int(self.color_H), int(self.color_W)
        d_resized = cv2.resize(d, (w_c, h_c), interpolation=cv2.INTER_NEAREST)

        tx, ty, tz = self.t_depth_to_color
        us_c, vs_c = np.meshgrid(np.arange(w_c), np.arange(h_c))

        xn = (us_c - self.cx_c) / self.fx_c
        yn = (vs_c - self.cy_c) / self.fy_c
        z = d_resized.copy()
        z[z <= 0.0] = 1.0

        xc = xn * z
        yc = yn * z
        zc = z

        xd = xc - tx
        yd = yc - ty
        zd = zc - tz
        zd[zd == 0.0] = 1e-6

        ud = (xd * self.fx_d / zd + self.cx_d).astype(np.float32)
        vd = (yd * self.fy_d / zd + self.cy_d).astype(np.float32)

        return cv2.remap(
            d,
            ud,
            vd,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def process_frame(self, rgb_img: np.ndarray, depth_img: np.ndarray) -> None:
        boxes: List = []
        labels: List[str] = []
        scores: List = []

        if self.enable_owlv2 and self.model is not None and self.processor is not None:
            try:
                img_pil = PILImage.fromarray(rgb_img)
                inputs = self.processor(text=self.text_labels, images=img_pil, return_tensors="pt").to(self.device)
                with self.torch.no_grad():
                    outputs = self.model(**inputs)
                target_sizes = self.torch.tensor([(img_pil.height, img_pil.width)]).to(self.device)
                result = self.processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=self.detection_threshold,
                    text_labels=self.text_labels,
                )[0]
                boxes = result["boxes"]
                scores = result["scores"]
                labels = result["text_labels"]
            except Exception as exc:
                self._warn_throttle("owlv2", 5.0, f"OWLv2 inference failed: {exc}")

        now = self.get_clock().now().to_msg()
        image_with_boxes = rgb_img.copy()
        marker_array = MarkerArray()

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            del score
            x0, y0, x1, y1 = [int(v) for v in box.tolist()]

            cv2.rectangle(image_with_boxes, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(
                image_with_boxes,
                str(label),
                (x0, max(y0 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            p_base = self.bbox_to_point(box.tolist(), depth_img)
            if p_base is None:
                continue

            p_base += np.array([-0.01, 0.0, 0.0], dtype=np.float64)
            pt = Point(x=float(p_base[0]), y=float(p_base[1]), z=float(p_base[2]))

            sphere = Marker()
            sphere.header.frame_id = "base_link"
            sphere.header.stamp = now
            sphere.ns = "detected_objects_3d"
            sphere.id = i
            sphere.type = Marker.SPHERE_LIST
            sphere.action = Marker.ADD
            sphere.scale.x = 0.05
            sphere.scale.y = 0.05
            sphere.scale.z = 0.05
            sphere.color.a = 1.0
            sphere.color.r = 1.0
            sphere.points.append(pt)

            text_m = Marker()
            text_m.header = sphere.header
            text_m.ns = "detected_texts"
            text_m.id = i
            text_m.type = Marker.TEXT_VIEW_FACING
            text_m.action = Marker.ADD
            text_m.pose.position.x = pt.x
            text_m.pose.position.y = pt.y
            text_m.pose.position.z = pt.z + 0.05
            text_m.pose.orientation.w = 1.0
            text_m.scale.z = 0.05
            text_m.color.a = 1.0
            text_m.color.r = 1.0
            text_m.color.g = 1.0
            text_m.color.b = 1.0
            text_m.text = str(label)
            marker_array.markers.extend([sphere, text_m])

            ps = PointStamped()
            ps.header.stamp = now
            ps.header.frame_id = "base_link"
            ps.point = pt
            self.object_pub.publish(ps)

        self.marker_pub.publish(marker_array)

        bgr = cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR)
        rgb_debug_msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        rgb_debug_msg.header.frame_id = "camera_color_frame"
        rgb_debug_msg.header.stamp = now
        self.image_pub.publish(rgb_debug_msg)

        depth_vis = depth_img.copy()
        valid = depth_vis > 0.0
        if valid.any():
            d_min = float(depth_vis[valid].min())
            d_max = float(depth_vis[valid].max())
            depth_vis[valid] = (depth_vis[valid] - d_min) / (d_max - d_min + 1e-6) * 255.0
        depth_u8 = depth_vis.astype(np.uint8)
        depth_bgr = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)

        for box, label in zip(boxes, labels):
            x0, y0, x1, y1 = [int(v) for v in box.tolist()]
            u_c = (x0 + x1) // 2
            v_c = (y0 + y1) // 2
            cv2.rectangle(depth_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.circle(depth_bgr, (u_c, v_c), 5, (0, 0, 255), -1)
            cv2.putText(depth_bgr, str(label), (x0, max(y0 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        depth_msg_out = self.bridge.cv2_to_imgmsg(depth_bgr, encoding="bgr8")
        depth_msg_out.header.frame_id = "camera_color_frame"
        depth_msg_out.header.stamp = now
        self.depth_pub.publish(depth_msg_out)

    def depth_at_pixel(self, depth_img: np.ndarray, u: int, v: int, kernel: int = 15) -> Optional[float]:
        h, w = depth_img.shape[:2]
        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))
        half = max(1, kernel // 2)
        patch = depth_img[max(v - half, 0) : min(v + half + 1, h), max(u - half, 0) : min(u + half + 1, w)]
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def bbox_to_point(self, bbox: List[float], depth_img: np.ndarray) -> Optional[np.ndarray]:
        if self.current_ee_pose is None:
            self._warn_throttle("missing_ee", 2.0, "No EE pose yet; cannot transform detections to base_link")
            return None

        x0, y0, x1, y1 = [int(v) for v in bbox]
        u = int((x0 + x1) * 0.5)
        v = int((y0 + y1) * 0.5)
        z = self.depth_at_pixel(depth_img, u, v)
        if z is None:
            return None

        x_cam = (u - self.cx_c) * z / self.fx_c
        y_cam = (v - self.cy_c) * z / self.fy_c
        p_cam = np.array([x_cam, y_cam, z], dtype=np.float64)

        r_ee_cam = quaternion_to_matrix(float(self.ee_q[0]), float(self.ee_q[1]), float(self.ee_q[2]), float(self.ee_q[3]))
        p_ee = r_ee_cam.T @ (p_cam - self.ee_t)

        tx, ty, tz, qx, qy, qz, qw = self.current_ee_pose
        r_base_ee = quaternion_to_matrix(qx, qy, qz, qw)
        p_base = np.array([tx, ty, tz], dtype=np.float64) + r_base_ee @ p_ee
        return p_base


def main(args=None) -> None:
    rclpy.init(args=args)
    node = KinovaWristCamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()