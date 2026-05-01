from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from interfaces.msg import TextureClassification
from interfaces.srv import ClassifyTexture, CaptureTactileImage


class TactileNode(Node):
    def __init__(self) -> None:
        super().__init__('tactile_node')
        self.latest_tactile_image: Optional[Image] = None
        self.latest_grasp_state: bool = False
        self._captured_images: dict[int, Image] = {}

        self.create_subscription(Image, '/gelsight/image_raw', self._gelsight_callback, 10)
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
        image = self._captured_images.get(request.object_id, self.latest_tactile_image)
        if image is None:
            response.success = False
            response.message = f'No tactile image available for object {request.object_id}.'
            response.texture_class = -1
            return response
        class_id = request.object_id % 3
        self._publish_texture_class(request.object_id, class_id)
        response.success = True
        response.message = f'Assigned stub texture class {class_id} to object {request.object_id}.'
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
        msg.note = 'Stub tactile classifier output.'
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
