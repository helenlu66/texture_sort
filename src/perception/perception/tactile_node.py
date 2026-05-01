from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from interfaces.msg import TextureClassification
from interfaces.srv import ClassifyTexture, CaptureTactileImage
from perception.tactile_classification_utils import classify_texture, load_refs, texture_scores


# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------

class TactileNode(Node):
    def __init__(self) -> None:
        super().__init__('tactile_node')
        self.latest_tactile_image: Optional[Image] = None
        self.latest_grasp_state: bool = False
        self._captured_images: dict[int, Image] = {}
        self._bridge = CvBridge()

        ref_dir = Path(__file__).parent
        self._refs = load_refs(ref_dir)
        counts = {cr.class_id: len(cr) for cr in self._refs}
        self.get_logger().info(f'Loaded reference images per class: {counts}')

        self.create_subscription(Image, '/gelsight/image_raw', self._gelsight_callback, qos_profile_sensor_data)
        self.create_subscription(Bool, '/grasp_state', self._grasp_state_callback, 10)
        self.texture_class_publisher = self.create_publisher(TextureClassification, '/texture_class', 10)
        self.create_service(ClassifyTexture, '/classify_texture', self.handle_classify_texture)
        self.create_service(CaptureTactileImage, '/capture_tactile_image', self._handle_capture)
        self.get_logger().info('tactile_node ready')

    def _handle_capture(self, request, response):
        """Snapshot the current gelsight frame and store it for the given object_id."""
        if self.latest_tactile_image is None:
            response.success = False
            response.message = 'No gelsight image available.'
            return response
        self._captured_images[request.object_id] = self.latest_tactile_image
        self.get_logger().info(f'Captured tactile image for object {request.object_id}.')
        response.success = True
        response.message = f'Captured tactile image for object {request.object_id}.'
        return response

    def handle_classify_texture(self, request, response):
        image_msg = self._captured_images.get(request.object_id, self.latest_tactile_image)
        if image_msg is None:
            response.success = False
            response.message = f'No tactile image available for object {request.object_id}.'
            response.texture_class = -1
            return response

        query_bgr = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        scores = texture_scores(query_bgr, self._refs)
        self.get_logger().info(
            'Texture classifier scores: ' +
            ', '.join(
                f'class {class_id}: combined={parts["combined"]:.4f}, '
                f'ncc={parts["ncc"]:.4f}, bv={parts["block_variance_score"]:.4f}, '
                f'coverage={parts["texture_coverage"]:.4f}, '
                f'query_coverage={parts["query_texture_coverage"]:.4f}'
                for class_id, parts in sorted(scores.items())
            )
        )
        class_id = classify_texture(query_bgr, self._refs)

        self.get_logger().info(f'Object {request.object_id} -> texture class {class_id}.')
        self._publish_texture_class(request.object_id, class_id)
        response.success = True
        response.message = f'Object {request.object_id} is texture class {class_id}.'
        response.texture_class = class_id
        return response

    def _gelsight_callback(self, msg: Image) -> None:
        self.latest_tactile_image = msg

    def _grasp_state_callback(self, msg: Bool) -> None:
        self.latest_grasp_state = msg.data

    def _publish_texture_class(self, object_id: int, class_id: int) -> None:
        msg = TextureClassification()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.object_id = object_id
        msg.texture_class = class_id
        msg.success = True
        msg.note = f'NCC+BV classifier — class {class_id}.'
        self.texture_class_publisher.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TactileNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
