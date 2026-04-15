from typing import Optional

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from texture_sorting_interfaces.msg import ObjectGrounding, ObjectGroundingArray
from texture_sorting_interfaces.srv import DetectObjectPose, InitializeSceneObjects

from .models import ObjectPoseEstimator, SceneRegistry


class VisionNode(Node):
    def __init__(self) -> None:
        super().__init__('vision_node')
        self.latest_rgb_image: Optional[Image] = None
        self.latest_depth_image: Optional[Image] = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self.scene_registry = SceneRegistry()
        self.pose_estimator = ObjectPoseEstimator()

        self.create_subscription(Image, '/camera/image_raw', self._rgb_callback, 10)
        self.create_subscription(Image, '/camera/depth/image_raw', self._depth_callback, 10)
        self.create_subscription(CameraInfo, '/camera/camera_info', self._camera_info_callback, 10)
        self.object_grounding_publisher = self.create_publisher(ObjectGroundingArray, '/objects', 10)
        self.create_service(InitializeSceneObjects, '/initialize_scene_objects', self.handle_initialize_scene_objects)
        self.create_service(DetectObjectPose, '/detect_object_pose', self.handle_detect_object_pose)
        self.get_logger().info('vision_node ready')

    def handle_initialize_scene_objects(self, request, response):
        del request
        response.object_ids = self.scene_registry.initialize_stub_scene(count=3)
        response.success = True
        response.message = 'Initialized stub scene registry.'
        self._publish_objects()
        return response

    def handle_detect_object_pose(self, request, response):
        try:
            pose_estimate = self.pose_estimator.estimate_pose_for_object(request.object_id, self.scene_registry)
            pose = Pose()
            pose.position.x = pose_estimate.x
            pose.position.y = pose_estimate.y
            pose.position.z = pose_estimate.z
            pose.orientation.w = 1.0
            response.pose = pose
            response.success = True
            response.message = f'Pose estimated for object {request.object_id}.'
        except KeyError:
            response.success = False
            response.message = f'Object {request.object_id} is not registered.'
        return response

    def _rgb_callback(self, msg: Image) -> None:
        self.latest_rgb_image = msg

    def _depth_callback(self, msg: Image) -> None:
        self.latest_depth_image = msg

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg

    def _publish_objects(self) -> None:
        msg = ObjectGroundingArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        for object_id, record in self.scene_registry.get_objects().items():
            grounding = ObjectGrounding()
            grounding.object_id = object_id
            grounding.pixel_x, grounding.pixel_y = record.pixel_xy
            msg.objects.append(grounding)
        self.object_grounding_publisher.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
