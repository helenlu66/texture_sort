from typing import Dict, List, Optional

import rclpy
from geometry_msgs.msg import Pose
from rclpy.action import ActionClient
from rclpy.node import Node
from std_srvs.srv import Trigger
from texture_sorting_interfaces.action import ExecuteGrasp, LoadObjectIntoBox
from texture_sorting_interfaces.msg import DeliveryState, TaskState, TextureClassification
from texture_sorting_interfaces.srv import CheckBaseReady, ClassifyTexture, DetectObjectPose, InitializeSceneObjects, StartSortAndDelivery


class TaskManagerNode(Node):
    def __init__(self) -> None:
        super().__init__('task_manager_node')
        self.state = 'idle'
        self.registered_object_ids: List[int] = []
        self.active_category: Optional[int] = None
        self.category_memory: Dict[int, int] = {}

        self.initialize_scene_client = self.create_client(InitializeSceneObjects, '/initialize_scene_objects')
        self.detect_object_pose_client = self.create_client(DetectObjectPose, '/detect_object_pose')
        self.classify_texture_client = self.create_client(ClassifyTexture, '/classify_texture')
        self.check_base_ready_client = self.create_client(CheckBaseReady, '/check_base_ready')
        self.execute_grasp_client = ActionClient(self, ExecuteGrasp, '/execute_grasp')
        self.load_object_into_box_client = ActionClient(self, LoadObjectIntoBox, '/load_object_into_box')
        self.create_subscription(TextureClassification, '/texture_class', self._texture_class_callback, 10)
        self.create_subscription(DeliveryState, '/delivery_state', self._delivery_state_callback, 10)
        self.task_state_publisher = self.create_publisher(TaskState, '/task_state', 10)
        self.active_category_publisher = self.create_publisher(TaskState, '/active_category', 10)
        self.create_service(StartSortAndDelivery, '/start_sort_and_delivery', self.handle_start_sort_and_delivery)
        self.start_delivery_trigger_client = self.create_client(Trigger, '/start_delivery_cycle')
        self.get_logger().info('task_manager_node ready')

    def handle_start_sort_and_delivery(self, request, response):
        del request
        try:
            self._run_scene_initialization()
            self._run_classification_loop()
            self._run_delivery_loop()
            response.accepted = True
            response.message = 'Stub sorting workflow completed.'
        except RuntimeError as exc:
            response.accepted = False
            response.message = str(exc)
        return response

    def _run_scene_initialization(self) -> None:
        self._update_state('initializing_scene')
        if not self.initialize_scene_client.wait_for_service(timeout_sec=1.0):
            raise RuntimeError('/initialize_scene_objects service unavailable.')
        future = self.initialize_scene_client.call_async(InitializeSceneObjects.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError('Scene initialization failed.')
        self.registered_object_ids = list(result.object_ids)

    def _run_classification_loop(self) -> None:
        self._update_state('classifying_objects')
        for object_id in self.registered_object_ids:
            if not self.detect_object_pose_client.wait_for_service(timeout_sec=1.0):
                raise RuntimeError('/detect_object_pose service unavailable.')
            pose_req = DetectObjectPose.Request()
            pose_req.object_id = object_id
            pose_future = self.detect_object_pose_client.call_async(pose_req)
            rclpy.spin_until_future_complete(self, pose_future)
            if not self.classify_texture_client.wait_for_service(timeout_sec=1.0):
                raise RuntimeError('/classify_texture service unavailable.')
            cls_req = ClassifyTexture.Request()
            cls_req.object_id = object_id
            cls_future = self.classify_texture_client.call_async(cls_req)
            rclpy.spin_until_future_complete(self, cls_future)
            cls_result = cls_future.result()
            if cls_result is None or not cls_result.success:
                raise RuntimeError(f'Classification failed for object {object_id}.')
            self.category_memory[object_id] = cls_result.texture_class

    def _run_delivery_loop(self) -> None:
        if not self.category_memory:
            raise RuntimeError('No classified objects available for delivery.')
        self.active_category = self._select_next_category()
        if self.active_category is None:
            raise RuntimeError('No active category available.')
        self._update_state('loading_active_category')
        for object_id, category_id in self.category_memory.items():
            if category_id != self.active_category:
                continue
            self._send_stub_grasp_goal(object_id)
            self._send_stub_load_goal(object_id, category_id)
        self._update_state('requesting_delivery')
        if self.start_delivery_trigger_client.wait_for_service(timeout_sec=1.0):
            trigger_future = self.start_delivery_trigger_client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, trigger_future)
        self._update_state('waiting_for_delivery_completion')

    def _send_stub_grasp_goal(self, object_id: int) -> None:
        if not self.execute_grasp_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warning('/execute_grasp action unavailable; skipping grasp call.')
            return
        goal = ExecuteGrasp.Goal()
        goal.object_id = object_id
        goal.target_pose = Pose()
        goal.target_pose.orientation.w = 1.0
        future = self.execute_grasp_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

    def _send_stub_load_goal(self, object_id: int, category_id: int) -> None:
        if not self.load_object_into_box_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warning('/load_object_into_box action unavailable; skipping load call.')
            return
        goal = LoadObjectIntoBox.Goal()
        goal.object_id = object_id
        goal.category_id = category_id
        future = self.load_object_into_box_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

    def _select_next_category(self) -> Optional[int]:
        return min(set(self.category_memory.values())) if self.category_memory else None

    def _update_state(self, state: str) -> None:
        self.state = state
        msg = TaskState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.state = state
        msg.active_category = -1 if self.active_category is None else self.active_category
        msg.pending_object_ids = self.registered_object_ids
        self.task_state_publisher.publish(msg)
        self.active_category_publisher.publish(msg)

    def _texture_class_callback(self, msg: TextureClassification) -> None:
        self.category_memory[msg.object_id] = msg.texture_class

    def _delivery_state_callback(self, msg: DeliveryState) -> None:
        if msg.state == 'at_loading' and self.state == 'waiting_for_delivery_completion':
            self._update_state('idle')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
