import time as _time
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

from interfaces.action import Delivery, ExecuteGrasp, GoToPrePose, LoadObjectIntoBox
from interfaces.msg import ObjectGroundingArray, ObjectRecord, TaskState
from std_srvs.srv import Trigger
from interfaces.srv import ClassifyTexture, StartSortAndDelivery

BOX_TAG_ID = 10


# Delivery order: class 3 objects are loaded and delivered first, then class 1.
# Class 2 objects are intentionally left undelivered.
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
        self.declare_parameter('object_detection_settle_sec', 3.0)
        self.declare_parameter('object_detection_timeout_sec', 30.0)
        self.declare_parameter('box_detection_settle_sec', 3.0)
        self.declare_parameter('arm_stabilize_sec', 1.0)
        self.declare_parameter('use_placeholder_delivery', False)

        groundings_topic = str(self.get_parameter('groundings_topic').value)
        self._ignored_ids = {int(i) for i in self.get_parameter('ignored_object_ids').value}
        self._action_timeout = float(self.get_parameter('action_timeout_sec').value)
        self._service_timeout = float(self.get_parameter('service_timeout_sec').value)
        self._detection_settle_sec = float(self.get_parameter('object_detection_settle_sec').value)
        self._detection_timeout_sec = float(self.get_parameter('object_detection_timeout_sec').value)
        self._box_detection_settle_sec = float(self.get_parameter('box_detection_settle_sec').value)
        self._arm_stabilize_sec = float(self.get_parameter('arm_stabilize_sec').value)
        self._use_placeholder_delivery = bool(self.get_parameter('use_placeholder_delivery').value)

        self._callback_group = ReentrantCallbackGroup()

        # object_id → ObjectRecord (texture_class, delivered)
        self._records: dict[int, ObjectRecord] = {}
        self._visible_grounding_ids: set[int] = set()

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
            self, Delivery, '/deliver', callback_group=self._callback_group)

        # Service clients
        self._classify = self.create_client(
            ClassifyTexture, '/classify_texture', callback_group=self._callback_group)
        self._open_gripper = self.create_client(
            Trigger, '/open_gripper', callback_group=self._callback_group)

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
        self._log_state('restart_requested')
        self._records.clear()
        self._last_box_position = None
        self._sm = self._build_sm()

        bb = Blackboard()
        bb.object_ids = []
        bb.all_object_ids = []
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
        sm.add_state('OPEN_GRIPPER',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_open_gripper),
            {OUTCOME_NEXT: 'GO_TO_PRE_GRASP_FOR_OBJECT_MAPPING', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('GO_TO_PRE_GRASP_FOR_OBJECT_MAPPING',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_go_to_pre_grasp),
            {OUTCOME_NEXT: 'STABILIZE_OBJECT_LOCATIONS_FOR_MAPPING', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('STABILIZE_OBJECT_LOCATIONS_FOR_MAPPING',
            CbState([OUTCOME_NEXT], self._stabilize_object_locations),
            {OUTCOME_NEXT: 'PREPARE_OBJECT_LIST'})

        sm.add_state('GO_TO_PRE_LOAD_FOR_BOX_MAPPING',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_go_to_pre_load),
            {OUTCOME_NEXT: 'STABILIZE_BOX_POSITION', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('STABILIZE_BOX_POSITION',
            CbState([OUTCOME_NEXT], self._stabilize_box_position),
            {OUTCOME_NEXT: 'GO_TO_PRE_GRASP_INITIAL'})

        sm.add_state('GO_TO_PRE_GRASP_INITIAL',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_go_to_pre_grasp),
            {OUTCOME_NEXT: 'STABILIZE_OBJECT_LOCATIONS_INITIAL', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('STABILIZE_OBJECT_LOCATIONS_INITIAL',
            CbState([OUTCOME_NEXT], self._stabilize_object_locations),
            {OUTCOME_NEXT: 'GRASP_THEN_RESET'})

        sm.add_state('PREPARE_OBJECT_LIST',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._prepare_object_list),
            {OUTCOME_NEXT: 'GO_TO_PRE_LOAD_FOR_BOX_MAPPING', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('GO_TO_PRE_GRASP',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_go_to_pre_grasp),
            {OUTCOME_NEXT: 'STABILIZE_OBJECT_LOCATIONS', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('STABILIZE_OBJECT_LOCATIONS',
            CbState([OUTCOME_NEXT], self._stabilize_object_locations),
            {OUTCOME_NEXT: 'GRASP_THEN_RESET'})

        sm.add_state('GRASP_THEN_RESET',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_grasp_then_reset),
            {OUTCOME_NEXT: 'CLASSIFY_TEXTURE', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('CLASSIFY_TEXTURE',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_classify_texture),
            {OUTCOME_NEXT: 'ADVANCE_CLASSIFICATION', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('ADVANCE_CLASSIFICATION',
            CbState([OUTCOME_MORE_OBJECTS, OUTCOME_NEXT], self._advance_classification),
            {OUTCOME_MORE_OBJECTS: 'GO_TO_PRE_GRASP', OUTCOME_NEXT: 'VERIFY_CLASSIFICATIONS'})

        sm.add_state('VERIFY_CLASSIFICATIONS',
            CbState([OUTCOME_NEXT, OUTCOME_MORE_OBJECTS, OUTCOME_FAILED], self._verify_classifications),
            {OUTCOME_NEXT: 'SETUP_BATCHES', OUTCOME_MORE_OBJECTS: 'GO_TO_PRE_GRASP', OUTCOME_FAILED: OUTCOME_FAILED})

        # --- Delivery phase ---
        sm.add_state('SETUP_BATCHES',
            CbState([OUTCOME_NEXT, OUTCOME_DONE, OUTCOME_FAILED], self._setup_batches),
            {OUTCOME_NEXT: 'START_BATCH', OUTCOME_DONE: OUTCOME_DONE, OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('START_BATCH',
            CbState([OUTCOME_NEXT, OUTCOME_DONE], self._start_batch),
            {OUTCOME_NEXT: 'LOAD_GO_TO_PRE_GRASP', OUTCOME_DONE: OUTCOME_DONE})

        sm.add_state('LOAD_GO_TO_PRE_GRASP',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_go_to_pre_grasp),
            {OUTCOME_NEXT: 'STABILIZE_OBJECT_LOCATIONS_AT_PRE_GRASP', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('STABILIZE_OBJECT_LOCATIONS_AT_PRE_GRASP',
            CbState([OUTCOME_NEXT], self._stabilize_object_locations),
            {OUTCOME_NEXT: 'PICKUP'})

        sm.add_state('PICKUP',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_pickup),
            {OUTCOME_NEXT: 'GO_TO_PRE_LOAD', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('GO_TO_PRE_LOAD',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_go_to_pre_load),
            {OUTCOME_NEXT: 'STABILIZE_BOX_POSITION_AT_PRE_LOAD', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('STABILIZE_BOX_POSITION_AT_PRE_LOAD',
            CbState([OUTCOME_NEXT], self._stabilize_box_position),
            {OUTCOME_NEXT: 'LOAD_OBJECT'})

        sm.add_state('LOAD_OBJECT',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_load_object),
            {OUTCOME_NEXT: 'ADVANCE_LOAD', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('ADVANCE_LOAD',
            CbState([OUTCOME_MORE_OBJECTS, OUTCOME_DELIVER], self._advance_load),
            {OUTCOME_MORE_OBJECTS: 'LOAD_GO_TO_PRE_GRASP', OUTCOME_DELIVER: 'GO_TO_PRE_LOAD_BEFORE_DELIVER'})

        sm.add_state('GO_TO_PRE_LOAD_BEFORE_DELIVER',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_go_to_pre_load),
            {OUTCOME_NEXT: 'DELIVER', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('DELIVER',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_deliver_unloading),
            {OUTCOME_NEXT: 'DELIVER_LOADING', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('DELIVER_LOADING',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_deliver_loading),
            {OUTCOME_NEXT: 'STABILIZE_BOX_POSITION_AFTER_RETURN', OUTCOME_FAILED: OUTCOME_FAILED})

        sm.add_state('STABILIZE_BOX_POSITION_AFTER_RETURN',
            CbState([OUTCOME_NEXT], self._stabilize_box_position),
            {OUTCOME_NEXT: 'ADVANCE_BATCH'})

        sm.add_state('ADVANCE_BATCH',
            CbState([OUTCOME_MORE_BATCHES, OUTCOME_DONE], self._advance_batch),
            {OUTCOME_MORE_BATCHES: 'START_BATCH', OUTCOME_DONE: 'GO_TO_PRE_GRASP_DONE'})

        sm.add_state('GO_TO_PRE_GRASP_DONE',
            CbState([OUTCOME_NEXT, OUTCOME_FAILED], self._state_go_to_pre_grasp),
            {OUTCOME_NEXT: OUTCOME_DONE, OUTCOME_FAILED: OUTCOME_FAILED})

        return sm

    # ------------------------------------------------------------------ #
    # Classification states
    # ------------------------------------------------------------------ #

    def _state_open_gripper(self, bb: Blackboard) -> str:
        self._log_state('open_gripper')
        if not self._open_gripper.wait_for_service(timeout_sec=self._service_timeout):
            bb.error = '/open_gripper service unavailable'
            return OUTCOME_FAILED
        future = self._open_gripper.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._service_timeout)
        res = future.result()
        if res is None or not res.success:
            bb.error = f'open_gripper failed: {getattr(res, "message", "no response")}'
            return OUTCOME_FAILED
        return OUTCOME_NEXT

    def _prepare_object_list(self, bb: Blackboard) -> str:
        self._log_state(
            f'prepare_object_list — waiting up to {self._detection_timeout_sec}s '
            'for /groundings'
        )
        wait_deadline = _time.monotonic() + self._detection_timeout_sec
        ids = []
        last_log = 0.0
        while _time.monotonic() < wait_deadline:
            ids = sorted(oid for oid in self._records if oid not in self._ignored_ids)
            if ids:
                break
            now = _time.monotonic()
            if now - last_log >= 2.0:
                self.get_logger().info('Waiting for object groundings...')
                last_log = now
            _time.sleep(0.1)
        if not ids:
            bb.error = f'No detected objects after {self._detection_timeout_sec:.1f}s (check /groundings).'
            return OUTCOME_FAILED

        settle_deadline = _time.monotonic() + self._detection_settle_sec
        while _time.monotonic() < settle_deadline:
            _time.sleep(0.1)
        ids = sorted(oid for oid in self._records if oid not in self._ignored_ids)
        if not ids:
            bb.error = 'No detected objects after settle wait (check /groundings).'
            return OUTCOME_FAILED
        bb.object_ids = ids
        bb.all_object_ids = ids
        bb.classify_idx = 0
        bb.current_object_id = ids[0]
        self.get_logger().info(f'Detected {len(ids)} objects (sorted by ID): {ids}')
        return OUTCOME_NEXT

    def _wait_for_arm_to_stabilize(self) -> None:
        if self._arm_stabilize_sec <= 0.0:
            return
        self.get_logger().info(f'Waiting {self._arm_stabilize_sec:.1f}s for arm to stabilize')
        _time.sleep(self._arm_stabilize_sec)

    def _stabilize_object_locations(self, bb: Blackboard) -> str:
        del bb
        self._log_state(
            f'stabilize_object_locations — waiting {self._detection_settle_sec}s '
            'for object groundings'
        )
        self._wait_for_arm_to_stabilize()
        deadline = _time.monotonic() + self._detection_settle_sec
        while _time.monotonic() < deadline:
            _time.sleep(0.1)
        ids = sorted(oid for oid in self._visible_grounding_ids if oid not in self._ignored_ids)
        if ids:
            self.get_logger().info(f'Refreshed visible object groundings at pre_grasp: {ids}')
        else:
            self.get_logger().warn('No object groundings visible after pre_grasp stabilization.')
        return OUTCOME_NEXT

    def _stabilize_box_position(self, bb: Blackboard) -> str:
        del bb
        self._log_state(
            f'stabilize_box_position — waiting {self._box_detection_settle_sec}s '
            f'for AprilTag {BOX_TAG_ID}'
        )
        self._wait_for_arm_to_stabilize()
        had_cache = self._last_box_position is not None
        deadline = _time.monotonic() + self._box_detection_settle_sec
        seen_box = BOX_TAG_ID in self._visible_grounding_ids
        while _time.monotonic() < deadline:
            seen_box = seen_box or BOX_TAG_ID in self._visible_grounding_ids
            _time.sleep(0.1)

        if seen_box and self._last_box_position is not None:
            p = self._last_box_position.point
            self.get_logger().info(
                f'Cached stabilized box position in base_link: '
                f'({p.x:.3f}, {p.y:.3f}, {p.z:.3f})'
            )
        elif had_cache and self._last_box_position is not None:
            p = self._last_box_position.point
            self.get_logger().warn(
                f'No live AprilTag {BOX_TAG_ID} while stabilizing; keeping cached box '
                f'position ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})'
            )
        else:
            self.get_logger().warn(
                f'No live or cached AprilTag {BOX_TAG_ID} position after stabilization.'
            )
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
            self.get_logger().warn(f'classify_texture failed for obj {oid}, skipping (will catch in verification)')
            return OUTCOME_NEXT
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

    def _verify_classifications(self, bb: Blackboard) -> str:
        self._log_state('verify_classifications')
        all_ids = list(bb.all_object_ids)
        missing = [oid for oid in all_ids if self._records.get(oid, ObjectRecord()).texture_class == -1]
        for oid in all_ids:
            rec = self._records.get(oid)
            cls = rec.texture_class if rec else -1
            self.get_logger().info(f'  obj {oid}: texture_class={cls}')
        if missing:
            self.get_logger().warn(f'Unclassified obj(s): {missing} — retrying grasp_then_reset')
            bb.object_ids = missing
            bb.classify_idx = 0
            bb.current_object_id = missing[0]
            return OUTCOME_MORE_OBJECTS
        summary = ', '.join(
            f'obj {oid} → class {self._records[oid].texture_class}' for oid in all_ids
        )
        self.get_logger().info(f'All objects classified: {summary}')
        return OUTCOME_NEXT

    # ------------------------------------------------------------------ #
    # Delivery states
    # ------------------------------------------------------------------ #

    def _setup_batches(self, bb: Blackboard) -> str:
        self._log_state('setup_batches')
        batches = []
        for cls in DELIVERY_ORDER:
            ids = [oid for oid in list(bb.all_object_ids)
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
        if BOX_TAG_ID in self._visible_grounding_ids:
            self.get_logger().info(
                f'AprilTag {BOX_TAG_ID} is currently visible; place_node will use live grounding.'
            )
        elif self._last_box_position is not None:
            goal.box_position = self._last_box_position
            p = self._last_box_position.point
            self.get_logger().info(
                f'AprilTag {BOX_TAG_ID} not currently visible; passing cached box position '
                f'in base_link: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})'
            )
        else:
            self.get_logger().warn(
                f'No live or cached AprilTag {BOX_TAG_ID} position; '
                'place_node will try its own live grounding fallback.'
            )
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

    def _state_deliver_unloading(self, bb: Blackboard) -> str:
        cls = DELIVERY_ORDER[int(bb.batch_idx)]
        self._log_state(f'deliver class {cls} objects {list(bb.loaded_this_batch)}')
        if self._use_placeholder_delivery:
            self.get_logger().info('[deliver] placeholder — waiting 5 s')
            _time.sleep(5.0)
            for oid in list(bb.loaded_this_batch):
                self._save_record(int(oid), delivered=True)
            return OUTCOME_NEXT
        goal = Delivery.Goal()
        goal.target = 'unloading'
        result = self._send_action(self._deliver, goal, '/deliver')
        if not self._ok(result):
            bb.error = self._msg(result, '/deliver failed')
            return OUTCOME_FAILED
        for oid in list(bb.loaded_this_batch):
            self._save_record(int(oid), delivered=True)
        return OUTCOME_NEXT

    def _state_deliver_loading(self, bb: Blackboard) -> str:
        cls = DELIVERY_ORDER[int(bb.batch_idx)]
        self._log_state(f'deliver_loading class {cls} — waiting 5 s after unloading')
        _time.sleep(5.0)
        if self._use_placeholder_delivery:
            self.get_logger().info('[deliver_loading] placeholder — skipping action')
            return OUTCOME_NEXT
        goal = Delivery.Goal()
        goal.target = 'loading'
        result = self._send_action(self._deliver, goal, '/deliver')
        if not self._ok(result):
            bb.error = self._msg(result, '/deliver loading failed')
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
        self._visible_grounding_ids = {g.object_id for g in msg.objects}
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
