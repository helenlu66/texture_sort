#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from apriltag_msgs.msg import AprilTagDetectionArray
from interfaces.msg import ObjectGrounding, ObjectGroundingArray
from cv_bridge import CvBridge
import cv2


class AprilTagOverlay(Node):
    def __init__(self):
        super().__init__('apriltag_overlay')
        self.bridge = CvBridge()
        self.latest_detections = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame_id = ''

        self.declare_parameter('image_topic', '/external_camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/external_camera/color/camera_info')
        self.declare_parameter('detections_topic', 'apriltag/detections')
        self.declare_parameter('overlay_topic', '/detections_image')
        self.declare_parameter('grounding_topic', '/groundings')
        self.declare_parameter('tag_size', 0.05)

        image_topic = str(self.get_parameter('image_topic').value)
        camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        detections_topic = str(self.get_parameter('detections_topic').value)
        overlay_topic = str(self.get_parameter('overlay_topic').value)
        grounding_topic = str(self.get_parameter('grounding_topic').value)
        self.tag_size = float(self.get_parameter('tag_size').value)

        h = self.tag_size / 2.0
        self.tag_corners_3d = np.array([
            [-h, -h, 0.0],
            [ h, -h, 0.0],
            [ h,  h, 0.0],
            [-h,  h, 0.0],
        ], dtype=np.float32)

        self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.create_subscription(CameraInfo, camera_info_topic, self._camera_info_callback, 10)
        self.create_subscription(AprilTagDetectionArray, detections_topic, self.detections_callback, 10)

        self.image_pub = self.create_publisher(Image, overlay_topic, 10)
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

        # Decompose the homography using camera intrinsics to get 3D pose.
        # The apriltag homography maps tag-plane coords (range -1..1) to image pixels.
        H = np.array(det.homography, dtype=np.float64).reshape(3, 3)
        K_inv = np.linalg.inv(self.camera_matrix)
        H_norm = K_inv @ H

        scale = (np.linalg.norm(H_norm[:, 0]) + np.linalg.norm(H_norm[:, 1])) / 2.0
        if scale < 1e-6:
            return None
        H_norm /= scale

        r1 = H_norm[:, 0]
        r2 = H_norm[:, 1]
        r3 = np.cross(r1, r2)
        rot_mat = np.column_stack([r1, r2, r3])
        U, _, Vt = np.linalg.svd(rot_mat)
        rot_mat = U @ Vt

        # Translation scaled from homography units (-1..1) to meters
        tvec = H_norm[:, 2] * (self.tag_size / 2.0)
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

        det_age = (
            (msg.header.stamp.sec - self.latest_detections.header.stamp.sec)
            + (msg.header.stamp.nanosec - self.latest_detections.header.stamp.nanosec) * 1e-9
        ) if self.latest_detections else 999

        if self.latest_detections and self.latest_detections.detections:
            for det in self.latest_detections.detections:
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
                    self.get_logger().info(f'Detected tag ID {det.id} at ({x:.2f}, {y:.2f}, {z:.2f}) m')
                    g = ObjectGrounding()
                    g.object_id = det.id
                    g.pose = pose
                    groundings.objects.append(g)
                    cv2.putText(frame, f'({x:.2f},{y:.2f},{z:.2f})m', (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    cv2.putText(frame, f'ID:{det.id} (no intrinsics)', (cx + 10, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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
