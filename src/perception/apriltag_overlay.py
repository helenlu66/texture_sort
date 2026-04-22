#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from apriltag_msgs.msg import AprilTagDetectionArray
from cv_bridge import CvBridge
import cv2

class AprilTagOverlay(Node):
    def __init__(self):
        super().__init__('apriltag_overlay')
        self.bridge = CvBridge()
        self.latest_detections = None

        self.sub_image = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)

        self.sub_detections = self.create_subscription(
            AprilTagDetectionArray,
            '/detections',
            self.detections_callback,
            10)

        self.pub = self.create_publisher(Image, '/detections_image', 10)
        self.get_logger().info('AprilTag overlay node started')

    def detections_callback(self, msg):
        self.latest_detections = msg

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if self.latest_detections:
            for det in self.latest_detections.detections:
                # Draw bounding box from corner points
                corners = [(int(c.x), int(c.y)) for c in det.corners]
                for i in range(4):
                    cv2.line(frame, corners[i], corners[(i + 1) % 4], (0, 255, 0), 2)

                # Draw center point
                cx, cy = int(det.centre.x), int(det.centre.y)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # Draw tag ID label
                label = f'ID: {det.id}'
                cv2.putText(frame, label, (corners[0][0], corners[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_msg.header = msg.header
        self.pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagOverlay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()