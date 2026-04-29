#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from apriltag_msgs.msg import AprilTagDetectionArray
from interfaces.msg import ObjectGrounding, ObjectGroundingArray
from cv_bridge import CvBridge
import cv2
import tf2_ros
import tf2_geometry_msgs


class AprilTagOverlay(Node):
    def __init__(self):
        super().__init__('apriltag_overlay')
        self.bridge = CvBridge()
        self.latest_detections = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame_id = ''

        self.declare_parameter('image_topic', '/wrist_camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/wrist_camera/color/camera_info')
        self.declare_parameter('detections_topic', 'apriltag/detections')
        self.declare_parameter('overlay_topic', '/detections_image')
        self.declare_parameter('grounding_topic', '/groundings')
        self.declare_parameter('tag_size', 0.05)
        self.declare_parameter('ee_frame', 'tool_frame')

        image_topic = str(self.get_parameter('image_topic').value)
        camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        detections_topic = str(self.get_parameter('detections_topic').value)
        overlay_topic = str(self.get_parameter('overlay_topic').value)
        grounding_topic = str(self.get_parameter('grounding_topic').value)
        self.tag_size = float(self.get_parameter('tag_size').value)
        self.ee_frame = str(self.get_parameter('ee_frame').value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        h = self.tag_size / 2.0
        self.tag_corners_3d = np.array([
            [-h, -h, 0.0],
            [ h, -h, 0.0],
            [ h,  h, 0.0],
            [-h,  h, 0.0],
        ], dtype=np.float32)

        self.create_subscription(Image, image_topic, self.image_callback, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, camera_info_topic, self._camera_info_callback, qos_profile_sensor_data)
        self.create_subscription(AprilTagDetectionArray, detections_topic, self.detections_callback, 10)

        self.image_pub = self.create_publisher(Image, overlay_topic, qos_profile_sensor_data)
        self.grounding_pub = self.create_publisher(ObjectGroundingArray, grounding_topic, 10)
        self.get_logger().info('AprilTag overlay node started')

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)
            self.camera_frame_id = msg.header.frame_id

    def detections_callback(self, msg):
        self.latest_detections = msg

    def get_grounding(self, det, stamp) -> PoseStamped | None:
        if self.camera_matrix is None:
            return None

        corners_2d = np.array([[c.x, c.y] for c in det.corners], dtype=np.float32)
        if corners_2d.shape != (4, 2):
            return None

        dist_coeffs = self.dist_coeffs
        if dist_coeffs is not None and dist_coeffs.size == 0:
            dist_coeffs = None

        ok, rvec, tvec = cv2.solvePnP(
            self.tag_corners_3d,
            corners_2d,
            self.camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

        rot_mat, _ = cv2.Rodrigues(rvec)
        q = _rot_to_quat(rot_mat)

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.camera_frame_id
        t = tvec.flatten()
        pose.pose.position.x = float(t[0])
        pose.pose.position.y = float(t[1])
        pose.pose.position.z = float(t[2])
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        return pose

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        groundings = ObjectGroundingArray()
        groundings.header = msg.header

        if self.latest_detections and self.latest_detections.detections:
            for det in self.latest_detections.detections:
                try:
                    corners = [(int(c.x), int(c.y)) for c in det.corners]
                    for i in range(4):
                        cv2.line(frame, corners[i], corners[(i + 1) % 4], (0, 255, 0), 2)

                    cx, cy = int(det.centre.x), int(det.centre.y)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                    pose = self.get_grounding(det, msg.header.stamp)
                    if pose:
                        x = pose.pose.position.x
                        y = pose.pose.position.y
                        z = pose.pose.position.z
                        g = ObjectGrounding()
                        g.object_id = det.id
                        g.pose = pose
                        groundings.objects.append(g)
                        cv2.putText(frame, f'CAM:({x:.2f},{y:.2f},{z:.2f})m', (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, f'ID:{det.id} (no intrinsics)', (cx + 10, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                except Exception as e:
                    self.get_logger().warn(f'skipping detection {det.id}: {e}', throttle_duration_sec=1.0)

        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_msg.header = msg.header
        self.image_pub.publish(out_msg)
        self.grounding_pub.publish(groundings)


def _rot_to_quat(r: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (r[2, 1] - r[1, 2]) * s
        y = (r[0, 2] - r[2, 0]) * s
        z = (r[1, 0] - r[0, 1]) * s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagOverlay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
