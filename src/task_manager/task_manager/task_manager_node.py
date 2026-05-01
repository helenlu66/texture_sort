from typing import Optional
import rclpy
import rclpy.duration
import rclpy.time
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from yasmin import Blackboard, CbState, StateMachine

from interfaces.action import Deliver, ExecuteGrasp, GoToPrePose, LoadObjectIntoBox, Return
from interfaces.msg import ObjectGroundingArray, ObjectRecord, TaskState
from interfaces.srv import ClassifyTexture, StartSortAndDelivery

BOX_TAG_ID = 10


# Delivery order: class 3 objects are loaded and delivered first, then class 1.
DELIVERY_ORDER = [3, 1]

OUTCOME_NEXT = 'next'
OUTCOME_MORE_OBJECTS = 'more_objects'
OUTCOME_MORE_BATCHES = 'more_batches'
OUTCOME_DELIVER = 'deliver'
OUTCOME_DONE = 'done'
OUTCOME_FAILED = 'failed'


class TaskManagerNode(Node):
    def __init__(self) -> None:
        super().__init__('task_manager_node')

        self.declare_parameter('groundings_topic', '/groundings')
        self.declare_parameter('ignored_object_ids', [10])
        self.declare_parameter('action_timeout_sec', 90.0)
        self.declare_parameter('service_timeout_sec', 10.0)

        groundings_topic = str(self.get_parameter('groundings_topic').value)
        self._ignored_ids = {int(i) for i in self.get_parameter('ignored_object_ids').value}
        self._action_timeout = float(self.get_parameter('action_timeout_sec').value)
        self._service_timeout = float(self.get_parameter('service_timeout_sec').value)

        self._callback_group = ReentrantCallbackGroup()

        # object_id → ObjectRecord (texture_class, delivered)
        self._records: dict[int, ObjectRecord] = {}

        # Last known box position in base_link, saved whenever tag 10 is seen.
        self._last_box_position: Optional[PointStamped] = None
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Action clients
        self._go_to_pre_grasp = ActionClient(
            self, GoToPrePose, '/go_to_pre_grasp', callback_group=self._callback_group)
        self._grasp_then_reset = ActionClient(
            self, ExecuteGrasp, '/grasp_then_reset', callback_group=self._callback_group)
        self._pickup = ActionClient(
            self, ExecuteGrasp, '/pickup', callback_group=self._callback_group)
        self._go_to_pre_load = ActionClient(
            self, GoToPrePose, '/go_to_pre_load', callback_group=self._callback_group)
        self._load_into_box = ActionClient(
            self, LoadObjectIntoBox, '/load_object_into_box', callback_group=self._callback_group)
        self._deliver = ActionClient(
            self, Deliver, '/deliver', callback_group=self._callback_group)
        self._return = ActionClient(
            self, Return, '/return', callback_group=self._callback_group)

        # Service clients
        self._classify = self.create_client(
            ClassifyTexture, '/classify_texture', callback_group=self._callback_group)

        self.create_subscription(
            ObjectGroundingArray, groundings_topic, self._groundings_callback, 10,
            callback_group=self._callback_group)

        self._task_state_pub = self.create_publisher(TaskState, '/task_state', 10)
        self._record_pub = self.create_publisher(ObjectRecord, '/object_records', 10)

        self.create_service(
            StartSortAndDelivery, '/start_classify_and_delivery',
            self._handle_start, callback_group=self._callback_group)

        self._sm = self._build_sm()
        self.get_logger().info('task_manager_node ready')

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #

    def _handle_start(self, request, response):
        del request
        for record in self._records.values():
            record.texture_class = -1
            record.classified = False
            record.delivered = False

        bb = Blackboard()
        bb.object_ids = []
        bb.classify_idx = 0
        bb.current_object_id = -1
        bb.batches = []      # list[list[int]], indexed by DELIVERY_ORDER
        bb.batch_idx = 0
        bb.load_idx = 0
        bb.loaded_this_batch = []
        bb.error = ''

        outcome = self._sm(bb)
        response.accepted = outcome == OUTCOME_DONE
        response.message = (
            f'Completed. batches={[list(b) for b in bb.batches]}'
            if outcome == OUTCOME_DONE
            else f'Failed: {bb.error}'
        )
        return response

    # ------------------------------------------------------------------ #
    # State machine
    # ------------------------------------------------------------------ #

    def _build_sm(self) -> StateMachine:
        sm = StateMachine(outcomes=[OUTCOME_DONE, OUTCOME_FAILED])

        # --- Classification phase ---
        sm.add_state('GO_TO_PRE_GRASP_INITIAL',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_go_to_pre_grasp),
            {OUTCOME_NEXT: 'PREPARE_OBJECT_LIST', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('PREPARE_OBJECT_LIST',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._prepare_object_list),
            {OUTCOME_NEXT: 'GRASP_THEN_RESET', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('GO_TO_PRE_GRASP',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_go_to_pre_grasp),
            {OUTCOME_NEXT: 'GRASP_THEN_RESET', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('GRASP_THEN_RESET',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_grasp_then_reset),
            {OUTCOME_NEXT: 'CLASSIFY_TEXTURE', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('CLASSIFY_TEXTURE',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_classify_texture),
            {OUTCOME_NEXT: 'ADVANCE_CLASSIFICATION', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('ADVANCE_CLASSIFICATION',
            CbState([OUTCOME_MORE_OBJECTS, OUTCOME_NEXT], self._advance_classification),
            {OUTCOME_MORE_OBJECTS: 'GO_TO_PRE_GRASP', OUTCOME_NEXT: 'SETUP_BATCHES'})

        # --- Delivery phase ---
        sm.add_state('SETUP_BATCHES',
            CbState([OUTCOME_NEXT, OUTCOME_DONE, OUTCOME_FAILED], self._setup_batches),
            {OUTCOME_NEXT: 'START_BATCH', OUTCOME_DONE: OUTCOME_DONE, OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('START_BATCH',
            CbState([OUTCOME_NEXT, OUTCOME_DONE], self._start_batch),
            {OUTCOME_NEXT: 'LOAD_GO_TO_PRE_GRASP', OUTCOME_DONE: OUTCOME_DONE})

        sm.add_state('LOAD_GO_TO_PRE_GRASP',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_go_to_pre_grasp),
            {OUTCOME_NEXT: 'PICKUP', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('PICKUP',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_pickup),
            {OUTCOME_NEXT: 'GO_TO_PRE_LOAD', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('GO_TO_PRE_LOAD',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_go_to_pre_load),
            {OUTCOME_NEXT: 'LOAD_OBJECT', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('LOAD_OBJECT',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_load_object),
            {OUTCOME_NEXT: 'ADVANCE_LOAD', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('ADVANCE_LOAD',
            CbState([OUTCOME_MORE_OBJECTS, OUTCOME_DELIVER], self._advance_load),
            {OUTCOME_MORE_OBJECTS: 'LOAD_GO_TO_PRE_GRASP', OUTCOME_DELIVER: 'DELIVER'})

        sm.add_state('DELIVER',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_deliver),
            {OUTCOME_NEXT: 'RETURN', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('RETURN',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_return),
            {OUTCOME_NEXT: 'ADVANCE_BATCH', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('ADVANCE_BATCH',
            CbState([OUTCOME_MORE_BATCHES, OUTCOME_DONE], self._advance_batch),
            {OUTCOME_MORE_BATCHES: 'START_BATCH', OUTCOME_DONE: OUTCOME_DONE})

        return sm

    # ------------------------------------------------------------------ #
    # Classification states
    # ------------------------------------------------------------------ #

    def _prepare_object_list(self, bb: Blackboard) -> str:
        self._log_state('prepare_object_list')
        ids = sorted(oid for oid in self._records if oid not in self._ignored_ids)
        if not ids:
            bb.error = 'No detected objects (check /groundings).'
            return OUTCOME_FAILED
        bb.object_ids = ids
        bb.classify_idx = 0
        bb.current_object_id = ids[0]
        self.get_logger().info(f'Objects to classify: {ids}')
        return OUTCOME_NEXT

    def _state_go_to_pre_grasp(self, bb: Blackboard) -> str:
        self._log_state(f'go_to_pre_grasp (obj {bb.current_object_id})')
        result = self._send_action(self._go_to_pre_grasp, GoToPrePose.Goal(), '/go_to_pre_grasp')
        if not self._ok(result):
            bb.error = self._msg(result, 'go_to_pre_grasp failed')
            return OUTCOME_FAILED
        return OUTCOME_NEXT

    def _state_grasp_then_reset(self, bb: Blackboard) -> str:
        oid = int(bb.current_object_id)
        self._log_state(f'grasp_then_reset obj {oid}')
        goal = ExecuteGrasp.Goal()
        goal.object_id = oid
        result = self._send_action(self._grasp_then_reset, goal, '/grasp_then_reset')
        if not self._ok(result):
            bb.error = self._msg(result, f'grasp_then_reset failed for obj {oid}')
            return OUTCOME_FAILED
        return OUTCOME_NEXT

    def _state_classify_texture(self, bb: Blackboard) -> str:
        oid = int(bb.current_object_id)
        self._log_state(f'classify_texture obj {oid}')
        if not self._classify.wait_for_service(timeout_sec=self._service_timeout):
            bb.error = '/classify_texture unavailable'
            return OUTCOME_FAILED
        req = ClassifyTexture.Request()
        req.object_id = oid
        future = self._classify.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._service_timeout)
        res = future.result()
        if res is None or not res.success:
            bb.error = f'classify_texture failed for obj {oid}'
            return OUTCOME_FAILED
        self._save_record(oid, texture_class=int(res.texture_class))
        self.get_logger().info(f'obj {oid} → texture class {res.texture_class}')
        return OUTCOME_NEXT

    def _advance_classification(self, bb: Blackboard) -> str:
        next_idx = int(bb.classify_idx) + 1
        bb.classify_idx = next_idx
        if next_idx < len(list(bb.object_ids)):
            bb.current_object_id = list(bb.object_ids)[next_idx]
            return OUTCOME_MORE_OBJECTS
        return OUTCOME_NEXT

    # ------------------------------------------------------------------ #
    # Delivery states
    # ------------------------------------------------------------------ #

    def _setup_batches(self, bb: Blackboard) -> str:
        self._log_state('setup_batches')
        batches = []
        for cls in DELIVERY_ORDER:
            ids = [oid for oid in list(bb.object_ids)
                   if self._records.get(oid) and self._records[oid].texture_class == cls]
            batches.append(ids)
            self.get_logger().info(f'Delivery batch class {cls}: {ids}')
        bb.batches = batches
        bb.batch_idx = 0
        if not any(batches):
            self._log_state('done')
            return OUTCOME_DONE
        return OUTCOME_NEXT

    def _start_batch(self, bb: Blackboard) -> str:
        batches = list(bb.batches)
        batch_idx = int(bb.batch_idx)
        # Find next non-empty batch
        while batch_idx < len(batches) and not batches[batch_idx]:
            batch_idx += 1
        if batch_idx >= len(batches):
            self._log_state('done')
            return OUTCOME_DONE
        bb.batch_idx = batch_idx
        batch = batches[batch_idx]
        bb.load_idx = 0
        bb.current_object_id = batch[0]
        bb.loaded_this_batch = []
        cls = DELIVERY_ORDER[batch_idx]
        self._log_state(f'start_batch class {cls}: {batch}')
        return OUTCOME_NEXT

    def _state_pickup(self, bb: Blackboard) -> str:
        oid = int(bb.current_object_id)
        self._log_state(f'pickup obj {oid}')
        goal = ExecuteGrasp.Goal()
        goal.object_id = oid
        result = self._send_action(self._pickup, goal, '/pickup')
        if not self._ok(result):
            bb.error = self._msg(result, f'pickup failed for obj {oid}')
            return OUTCOME_FAILED
        return OUTCOME_NEXT

    def _state_go_to_pre_load(self, bb: Blackboard) -> str:
        self._log_state(f'go_to_pre_load (obj {bb.current_object_id})')
        result = self._send_action(self._go_to_pre_load, GoToPrePose.Goal(), '/go_to_pre_load')
        if not self._ok(result):
            bb.error = self._msg(result, 'go_to_pre_load failed')
            return OUTCOME_FAILED
        return OUTCOME_NEXT

    def _state_load_object(self, bb: Blackboard) -> str:
        oid = int(bb.current_object_id)
        self._log_state(f'load_object_into_box obj {oid}')
        goal = LoadObjectIntoBox.Goal()
        goal.object_id = oid
        if self._last_box_position is not None:
            goal.box_position = self._last_box_position
            p = self._last_box_position.point
            self.get_logger().info(
                f'Passing last known box position in base_link: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})'
            )
        else:
            self.get_logger().warn('No cached box position — place_node will fall back to AprilTag detection.')
        result = self._send_action(self._load_into_box, goal, '/load_object_into_box')
        if not self._ok(result):
            bb.error = self._msg(result, f'load_object_into_box failed for obj {oid}')
            return OUTCOME_FAILED
        loaded = list(bb.loaded_this_batch)
        loaded.append(oid)
        bb.loaded_this_batch = loaded
        return OUTCOME_NEXT

    def _advance_load(self, bb: Blackboard) -> str:
        batch = list(bb.batches)[int(bb.batch_idx)]
        next_idx = int(bb.load_idx) + 1
        bb.load_idx = next_idx
        if next_idx < len(batch):
            bb.current_object_id = batch[next_idx]
            return OUTCOME_MORE_OBJECTS
        return OUTCOME_DELIVER

    def _state_deliver(self, bb: Blackboard) -> str:
        self._log_state(f'deliver batch {list(bb.loaded_this_batch)}')
        result = self._send_action(self._deliver, Deliver.Goal(), '/deliver')
        if not self._ok(result):
            bb.error = self._msg(result, '/deliver failed')
            return OUTCOME_FAILED
        for oid in list(bb.loaded_this_batch):
            self._save_record(int(oid), delivered=True)
        return OUTCOME_NEXT

    def _state_return(self, bb: Blackboard) -> str:
        self._log_state('return')
        result = self._send_action(self._return, Return.Goal(), '/return')
        if not self._ok(result):
            bb.error = self._msg(result, '/return failed')
            return OUTCOME_FAILED
        return OUTCOME_NEXT

    def _advance_batch(self, bb: Blackboard) -> str:
        bb.batch_idx = int(bb.batch_idx) + 1
        if int(bb.batch_idx) < len(list(bb.batches)):
            return OUTCOME_MORE_BATCHES
        self._log_state('done')
        self._log_records()
        return OUTCOME_DONE

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _send_action(self, client: ActionClient, goal, name: str):
        if not client.wait_for_server(timeout_sec=self._service_timeout):
            self.get_logger().error(f'{name} server unavailable')
            return None
        gf = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, gf, timeout_sec=self._service_timeout)
        gh = gf.result()
        if gh is None or not gh.accepted:
            self.get_logger().error(f'{name} goal rejected')
            return None
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf, timeout_sec=self._action_timeout)
        r = rf.result()
        if r is None:
            self.get_logger().error(f'{name} timed out')
            return None
        return r.result

    def _ok(self, result) -> bool:
        return result is not None and bool(getattr(result, 'success', False))

    def _msg(self, result, fallback: str) -> str:
        return str(getattr(result, 'message', fallback)) if result is not None else fallback

    def _ensure_record(self, oid: int) -> ObjectRecord:
        if oid not in self._records:
            r = ObjectRecord()
            r.object_id = oid
            r.texture_class = -1
            r.classified = False
            r.delivered = False
            self._records[oid] = r
        return self._records[oid]

    def _save_record(self, oid: int, texture_class: Optional[int] = None, delivered: Optional[bool] = None) -> None:
        r = self._ensure_record(oid)
        if texture_class is not None:
            r.texture_class = texture_class
            r.classified = True
        if delivered is not None:
            r.delivered = delivered
        self._record_pub.publish(r)

    def _log_state(self, state: str) -> None:
        msg = TaskState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.state = state
        self._task_state_pub.publish(msg)
        self.get_logger().info(f'[task] {state}')

    def _log_records(self) -> None:
        for oid in sorted(self._records):
            if oid in self._ignored_ids:
                continue
            r = self._records[oid]
            self.get_logger().info(
                f'  obj {r.object_id}: class={r.texture_class} classified={r.classified} delivered={r.delivered}')

    def _groundings_callback(self, msg: ObjectGroundingArray) -> None:
        for g in msg.objects:
            self._ensure_record(g.object_id)
            if g.object_id == BOX_TAG_ID:
                try:
                    tf = self._tf_buffer.lookup_transform(
                        'base_link',
                        g.pose.header.frame_id,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.1),
                    )
                    pose_in_base = tf2_geometry_msgs.do_transform_pose_stamped(g.pose, tf)
                    pt = PointStamped()
                    pt.header.frame_id = 'base_link'
                    pt.header.stamp = self.get_clock().now().to_msg()
                    pt.point = pose_in_base.pose.position
                    self._last_box_position = pt
                except Exception:
                    pass  # keep previous value if TF not ready yet


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
