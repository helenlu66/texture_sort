from typing import Dict, List, Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger
from yasmin import Blackboard, CbState, StateMachine

from interfaces.action import ExecuteGrasp, GoToPrePose, LoadObjectIntoBox
from interfaces.msg import (
    DeliveryState,
    ObjectGroundingArray,
    ObjectRecord,
    TaskState,
    TextureClassification,
)
from interfaces.srv import ClassifyTexture, StartSortAndDelivery


OUTCOME_NEXT = 'next'
OUTCOME_CLASS_1 = 'class_1'
OUTCOME_OTHER_CLASS = 'other_class'
OUTCOME_MORE_OBJECTS = 'more_objects'
OUTCOME_DONE = 'done'
OUTCOME_FAILED = 'failed'


class TaskManagerNode(Node):
    def __init__(self) -> None:
        super().__init__('task_manager_node')

        self.declare_parameter('groundings_topic', '/groundings')
        self.declare_parameter('target_texture_class', 1)
        self.declare_parameter('ignored_object_ids', [10])
        self.declare_parameter('action_timeout_sec', 90.0)
        self.declare_parameter('service_timeout_sec', 10.0)

        groundings_topic = str(self.get_parameter('groundings_topic').value)
        self._target_texture_class = int(self.get_parameter('target_texture_class').value)
        self._ignored_object_ids = {
            int(object_id) for object_id in self.get_parameter('ignored_object_ids').value
        }
        self._action_timeout_sec = float(self.get_parameter('action_timeout_sec').value)
        self._service_timeout_sec = float(self.get_parameter('service_timeout_sec').value)

        self.state = 'idle'
        self.registered_object_ids: List[int] = []
        self.active_category: Optional[int] = self._target_texture_class
        self.object_records: Dict[int, ObjectRecord] = {}
        self._callback_group = ReentrantCallbackGroup()

        self.go_to_pre_grasp_client = ActionClient(
            self, GoToPrePose, '/go_to_pre_grasp', callback_group=self._callback_group
        )
        self.execute_grasp_client = ActionClient(
            self, ExecuteGrasp, '/execute_grasp', callback_group=self._callback_group
        )
        self.load_object_into_box_client = ActionClient(
            self, LoadObjectIntoBox, '/load_object_into_box', callback_group=self._callback_group
        )
        self.reset_grasp_client = self.create_client(
            Trigger, '/reset_grasp', callback_group=self._callback_group
        )
        self.classify_texture_client = self.create_client(
            ClassifyTexture, '/classify_texture', callback_group=self._callback_group
        )
        self.start_delivery_trigger_client = self.create_client(
            Trigger, '/start_delivery_cycle', callback_group=self._callback_group
        )

        self.create_subscription(
            ObjectGroundingArray, groundings_topic, self._groundings_callback, 10,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            TextureClassification, '/texture_class', self._texture_class_callback, 10,
            callback_group=self._callback_group,
        )
        self.create_subscription(
            DeliveryState, '/delivery_state', self._delivery_state_callback, 10,
            callback_group=self._callback_group,
        )

        self.task_state_publisher = self.create_publisher(TaskState, '/task_state', 10)
        self.active_category_publisher = self.create_publisher(TaskState, '/active_category', 10)
        self.create_service(
            StartSortAndDelivery,
            '/start_sort_and_delivery',
            self.handle_start_sort_and_delivery,
            callback_group=self._callback_group,
        )

        self._sm = self._build_state_machine()
        self.get_logger().info('task_manager_node ready')

    def handle_start_sort_and_delivery(self, request, response):
        del request
        blackboard = Blackboard()
        blackboard.object_ids = []
        blackboard.current_index = 0
        blackboard.current_object_id = -1
        blackboard.error = ''

        outcome = self._sm(blackboard)
        response.accepted = outcome == OUTCOME_DONE
        response.message = (
            'Sort and load workflow completed.'
            if outcome == OUTCOME_DONE
            else f'Sort and load workflow failed: {blackboard.error}'
        )
        return response

    def _build_state_machine(self) -> StateMachine:
        sm = StateMachine(outcomes=[OUTCOME_DONE, OUTCOME_FAILED])
        sm.add_state(
            'PREPARE_OBJECT_LIST',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._prepare_object_list),
            transitions={OUTCOME_NEXT: 'GO_TO_PRE_GRASP', OUTCOME_FAILED: OUTCOME_FAILED},
        )
        sm.add_state(
            'GO_TO_PRE_GRASP',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._go_to_pre_grasp),
            transitions={OUTCOME_NEXT: 'EXECUTE_GRASP', OUTCOME_FAILED: OUTCOME_FAILED},
        )
        sm.add_state(
            'EXECUTE_GRASP',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._execute_grasp),
            transitions={OUTCOME_NEXT: 'CLASSIFY_TEXTURE', OUTCOME_FAILED: OUTCOME_FAILED},
        )
        sm.add_state(
            'CLASSIFY_TEXTURE',
            CbState([OUTCOME_CLASS_1, OUTCOME_OTHER_CLASS, OUTCOME_FAILED], self._classify_texture),
            transitions={
                OUTCOME_CLASS_1: 'LOAD_OBJECT_INTO_BOX',
                OUTCOME_OTHER_CLASS: 'RESET_NON_TARGET_OBJECT',
                OUTCOME_FAILED: OUTCOME_FAILED,
            },
        )
        sm.add_state(
            'LOAD_OBJECT_INTO_BOX',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._load_object_into_box),
            transitions={OUTCOME_NEXT: 'ADVANCE_OBJECT', OUTCOME_FAILED: OUTCOME_FAILED},
        )
        sm.add_state(
            'RESET_NON_TARGET_OBJECT',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._reset_non_target_object),
            transitions={OUTCOME_NEXT: 'ADVANCE_OBJECT', OUTCOME_FAILED: OUTCOME_FAILED},
        )
        sm.add_state(
            'ADVANCE_OBJECT',
            CbState([OUTCOME_MORE_OBJECTS, OUTCOME_DONE], self._advance_object),
            transitions={OUTCOME_MORE_OBJECTS: 'GO_TO_PRE_GRASP', OUTCOME_DONE: OUTCOME_DONE},
        )
        return sm

    def _prepare_object_list(self, blackboard: Blackboard) -> str:
        self._update_state('preparing_object_list')
        object_ids = sorted(
            object_id for object_id in self.object_records
            if object_id not in self._ignored_object_ids
        )
        if not object_ids:
            blackboard.error = 'No detected object IDs available from /groundings.'
            return OUTCOME_FAILED

        self.registered_object_ids = object_ids
        blackboard.object_ids = object_ids
        blackboard.current_index = 0
        blackboard.current_object_id = object_ids[0]
        self.get_logger().info(f'Task object order: {object_ids}')
        return OUTCOME_NEXT

    def _go_to_pre_grasp(self, blackboard: Blackboard) -> str:
        object_id = int(blackboard.current_object_id)
        self._update_state(f'go_to_pre_grasp_object_{object_id}')
        goal = GoToPrePose.Goal()
        result = self._send_action_goal(self.go_to_pre_grasp_client, goal, '/go_to_pre_grasp')
        if not self._action_result_success(result):
            blackboard.error = self._action_result_message(result, 'go_to_pre_grasp failed')
            return OUTCOME_FAILED
        return OUTCOME_NEXT

    def _execute_grasp(self, blackboard: Blackboard) -> str:
        object_id = int(blackboard.current_object_id)
        self._update_state(f'execute_grasp_object_{object_id}')
        goal = ExecuteGrasp.Goal()
        goal.object_id = object_id
        result = self._send_action_goal(self.execute_grasp_client, goal, '/execute_grasp')
        if not self._action_result_success(result):
            blackboard.error = self._action_result_message(result, f'execute_grasp failed for object {object_id}')
            return OUTCOME_FAILED
        return OUTCOME_NEXT

    def _classify_texture(self, blackboard: Blackboard) -> str:
        object_id = int(blackboard.current_object_id)
        self._update_state(f'classify_texture_object_{object_id}')
        if not self.classify_texture_client.wait_for_service(timeout_sec=self._service_timeout_sec):
            blackboard.error = '/classify_texture service unavailable.'
            return OUTCOME_FAILED

        request = ClassifyTexture.Request()
        request.object_id = object_id
        future = self.classify_texture_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._service_timeout_sec)
        result = future.result()
        if result is None or not result.success:
            blackboard.error = f'Classification failed for object {object_id}.'
            return OUTCOME_FAILED

        record = self._ensure_record(object_id)
        record.texture_class = int(result.texture_class)
        record.classified = True
        record.delivered = False
        self.get_logger().info(
            f'Object {object_id} classified as texture class {record.texture_class}.'
        )

        if record.texture_class == self._target_texture_class:
            return OUTCOME_CLASS_1
        return OUTCOME_OTHER_CLASS

    def _load_object_into_box(self, blackboard: Blackboard) -> str:
        object_id = int(blackboard.current_object_id)
        self._update_state(f'load_object_{object_id}_into_box')
        goal = LoadObjectIntoBox.Goal()
        goal.object_id = object_id
        result = self._send_action_goal(
            self.load_object_into_box_client,
            goal,
            '/load_object_into_box',
        )
        if not self._action_result_success(result):
            blackboard.error = self._action_result_message(
                result,
                f'load_object_into_box failed for object {object_id}',
            )
            return OUTCOME_FAILED

        self._ensure_record(object_id).delivered = True
        return OUTCOME_NEXT

    def _reset_non_target_object(self, blackboard: Blackboard) -> str:
        object_id = int(blackboard.current_object_id)
        self._update_state(f'reset_non_target_object_{object_id}')
        if not self.reset_grasp_client.wait_for_service(timeout_sec=self._service_timeout_sec):
            blackboard.error = '/reset_grasp service unavailable.'
            return OUTCOME_FAILED

        future = self.reset_grasp_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._service_timeout_sec)
        result = future.result()
        if result is None or not result.success:
            blackboard.error = f'reset_grasp failed after classifying object {object_id}.'
            return OUTCOME_FAILED

        self._ensure_record(object_id).delivered = False
        return OUTCOME_NEXT

    def _advance_object(self, blackboard: Blackboard) -> str:
        object_ids = list(blackboard.object_ids)
        next_index = int(blackboard.current_index) + 1
        blackboard.current_index = next_index
        if next_index >= len(object_ids):
            self._update_state('done')
            self._log_object_records()
            return OUTCOME_DONE

        blackboard.current_object_id = object_ids[next_index]
        return OUTCOME_MORE_OBJECTS

    def _send_action_goal(self, client: ActionClient, goal, action_name: str):
        if not client.wait_for_server(timeout_sec=self._service_timeout_sec):
            self.get_logger().error(f'{action_name} action server unavailable.')
            return None

        goal_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future, timeout_sec=self._service_timeout_sec)
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f'{action_name} goal rejected.')
            return None

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self._action_timeout_sec)
        result_msg = result_future.result()
        if result_msg is None:
            self.get_logger().error(f'{action_name} result timed out.')
            return None
        return result_msg.result

    def _action_result_success(self, result) -> bool:
        return result is not None and bool(getattr(result, 'success', False))

    def _action_result_message(self, result, fallback: str) -> str:
        if result is None:
            return fallback
        return str(getattr(result, 'message', fallback))

    def _ensure_record(self, object_id: int) -> ObjectRecord:
        record = self.object_records.get(object_id)
        if record is None:
            record = ObjectRecord()
            record.object_id = object_id
            record.texture_class = -1
            record.classified = False
            record.delivered = False
            self.object_records[object_id] = record
        return record

    def _update_state(self, state: str) -> None:
        self.state = state
        msg = TaskState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.state = state
        msg.active_category = self._target_texture_class
        msg.pending_object_ids = [
            object_id for object_id in self.registered_object_ids
            if not self.object_records.get(object_id, ObjectRecord()).delivered
        ]
        self.task_state_publisher.publish(msg)
        self.active_category_publisher.publish(msg)
        self.get_logger().info(f'[task] {state}')

    def _log_object_records(self) -> None:
        for object_id in sorted(self.object_records):
            if object_id in self._ignored_object_ids:
                continue
            record = self.object_records[object_id]
            self.get_logger().info(
                f'ObjectRecord object_id={record.object_id}, '
                f'texture_class={record.texture_class}, '
                f'classified={record.classified}, delivered={record.delivered}'
            )

    def _groundings_callback(self, msg: ObjectGroundingArray) -> None:
        for grounding in msg.objects:
            record = self._ensure_record(grounding.object_id)
            record.pose = grounding.pose.pose

    def _texture_class_callback(self, msg: TextureClassification) -> None:
        record = self._ensure_record(msg.object_id)
        record.texture_class = msg.texture_class
        record.classified = msg.success

    def _delivery_state_callback(self, msg: DeliveryState) -> None:
        if msg.state == 'at_loading' and self.state == 'waiting_for_delivery_completion':
            self._update_state('idle')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskManagerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
