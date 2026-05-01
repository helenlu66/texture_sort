from typing import Optional
import json
from pathlib import Path
import threading

from ament_index_python.packages import get_package_share_directory
import rclpy
import rclpy.duration
import rclpy.time
from rclpy.action import ActionServer, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import (
    PoseStamped, Quaternion, Vector3,
)
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints,
    PositionConstraint, OrientationConstraint, JointConstraint,
    BoundingVolume, WorkspaceParameters,
    MoveItErrorCodes,
)
from control_msgs.action import GripperCommand
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

from interfaces.action import GoToPrePose, LoadObjectIntoBox
from interfaces.msg import ObjectGroundingArray

BOX_TAG_ID = 10


class PlaceNode(Node):
    def __init__(self) -> None:
        super().__init__('place_node')

        self.declare_parameter('groundings_topic', '/groundings')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('ee_link', 'end_effector_link')
        self.declare_parameter('move_group', 'manipulator')
        self.declare_parameter('vel_scale', 0.3)
        self.declare_parameter('lower_vel_scale', 0.1)
        self.declare_parameter('accel_scale', 0.3)
        self.declare_parameter('loading_height', 0.10)
        self.declare_parameter('gripper_open_position', 0.0)
        self.declare_parameter('grasp_point_offset_x', 0.0)
        self.declare_parameter('grasp_point_offset_y', 0.0)
        self.declare_parameter('grasp_point_offset_z', 0.135)
        self.declare_parameter('pre_load_path', '')

        groundings_topic      = str(self.get_parameter('groundings_topic').value)
        self._base_frame      = str(self.get_parameter('base_frame').value)
        self._ee_link         = str(self.get_parameter('ee_link').value)
        self._move_group      = str(self.get_parameter('move_group').value)
        self._vel_scale       = float(self.get_parameter('vel_scale').value)
        self._lower_vel_scale = float(self.get_parameter('lower_vel_scale').value)
        self._accel_scale     = float(self.get_parameter('accel_scale').value)
        self._loading_height  = float(self.get_parameter('loading_height').value)
        self._gripper_open    = float(self.get_parameter('gripper_open_position').value)
        self._grasp_point_offset = Vector3(
            x=float(self.get_parameter('grasp_point_offset_x').value),
            y=float(self.get_parameter('grasp_point_offset_y').value),
            z=float(self.get_parameter('grasp_point_offset_z').value),
        )
        self._pre_load_path   = str(self.get_parameter('pre_load_path').value)
        self._action_callback_group = ReentrantCallbackGroup()

        self.current_joint_state: Optional[JointState] = None
        self.tf_buffer  = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._object_poses: dict[int, PoseStamped] = {}

        self.create_subscription(JointState, '/joint_states', self._joint_state_callback, 10)
        self.create_subscription(ObjectGroundingArray, groundings_topic, self._groundings_callback, 10)

        self._move_group_client = ActionClient(
            self, MoveGroup, '/move_action',
            callback_group=self._action_callback_group,
        )
        self._gripper_client = ActionClient(
            self, GripperCommand, '/robotiq_gripper_controller/gripper_cmd',
            callback_group=self._action_callback_group,
        )
        ActionServer(
            self, GoToPrePose, '/go_to_pre_load',
            self._go_to_pre_load_callback,
            callback_group=self._action_callback_group,
        )
        ActionServer(
            self, LoadObjectIntoBox, '/load_object_into_box',
            self._load_object_into_box_callback,
            callback_group=self._action_callback_group,
        )

        self.get_logger().info('place_node ready')

    # ------------------------------------------------------------------ #
    # Subscriptions                                                        #
    # ------------------------------------------------------------------ #

    def _groundings_callback(self, msg: ObjectGroundingArray) -> None:
        for g in msg.objects:
            self._object_poses[g.object_id] = g.pose

    def _joint_state_callback(self, msg: JointState) -> None:
        self.current_joint_state = msg

    # ------------------------------------------------------------------ #
    # Actions                                                              #
    # ------------------------------------------------------------------ #

    def _go_to_pre_load_callback(self, goal_handle):
        def fb(state: str):
            msg = GoToPrePose.Feedback()
            msg.state = state
            goal_handle.publish_feedback(msg)
            self.get_logger().info(f'[go_to_pre_load] {state}')

        fb('loading_pre_load_pose')
        pre_load_path = self._resolve_pre_load_path()
        try:
            with pre_load_path.open('r', encoding='utf-8') as f:
                pre_load = json.load(f)
        except Exception as e:
            goal_handle.abort()
            r = GoToPrePose.Result()
            r.success = False
            r.message = f'Failed to read {pre_load_path}: {e}'
            return r

        ee_pose_data = pre_load.get('ee_pose')
        arm_joints = pre_load.get('arm_joints')

        fb('moving_to_pre_load')
        if ee_pose_data:
            p = ee_pose_data.get('position', {})
            o = ee_pose_data.get('orientation', {})
            target = PoseStamped()
            target.header.frame_id = pre_load.get('frame_id', self._base_frame)
            target.pose.position.x = float(p.get('x', 0.0))
            target.pose.position.y = float(p.get('y', 0.0))
            target.pose.position.z = float(p.get('z', 0.0))
            target.pose.orientation.x = float(o.get('x', 0.0))
            target.pose.orientation.y = float(o.get('y', 0.0))
            target.pose.orientation.z = float(o.get('z', 0.0))
            target.pose.orientation.w = float(o.get('w', 1.0))
            self.get_logger().info(
                f'Moving to pre_load (pose goal): '
                f'pos=({target.pose.position.x:.4f}, {target.pose.position.y:.4f}, {target.pose.position.z:.4f})'
            )
            ok, message = self._move_to_pose(target)
        elif isinstance(arm_joints, dict) and arm_joints:
            self.get_logger().info(
                'Moving to pre_load (joint goal): '
                + ', '.join(f'{n}={v:.4f}' for n, v in arm_joints.items())
            )
            ok, message = self._move_to_joint_positions(arm_joints, velocity_scale=self._lower_vel_scale)
        else:
            goal_handle.abort()
            r = GoToPrePose.Result()
            r.success = False
            r.message = f'No ee_pose or arm_joints in {pre_load_path}'
            return r
        if not ok:
            goal_handle.abort()
            r = GoToPrePose.Result()
            r.success = False
            r.message = f'Move to pre_load failed: {message}'
            return r

        fb('done')
        goal_handle.succeed()
        r = GoToPrePose.Result()
        r.success = True
        r.message = 'Reached pre_load position.'
        return r

    def _load_object_into_box_callback(self, goal_handle):
        def fb(state: str):
            msg = LoadObjectIntoBox.Feedback()
            msg.state = state
            goal_handle.publish_feedback(msg)
            self.get_logger().info(f'[load_object_into_box] {state}')

        fb('resolving_box_pose')
        goal_box_position = goal_handle.request.box_position
        if goal_box_position.header.frame_id:
            # Caller provided an explicit box position — transform to base_link if needed.
            if goal_box_position.header.frame_id != self._base_frame:
                try:
                    tf = self.tf_buffer.lookup_transform(
                        self._base_frame,
                        goal_box_position.header.frame_id,
                        goal_box_position.header.stamp,
                        timeout=rclpy.duration.Duration(seconds=1.0),
                    )
                except Exception as e:
                    return self._abort(goal_handle, f'TF lookup for provided box_position failed: {e}')
                from geometry_msgs.msg import PointStamped
                pt_in_base = tf2_geometry_msgs.do_transform_point(goal_box_position, tf)
            else:
                pt_in_base = goal_box_position

            box_in_base = PoseStamped()
            box_in_base.header.frame_id = self._base_frame
            box_in_base.pose.position = pt_in_base.point
            self.get_logger().info(
                f'[load_object_into_box] using provided box_position in base_link: '
                f'pos=({pt_in_base.point.x:.4f}, {pt_in_base.point.y:.4f}, {pt_in_base.point.z:.4f})'
            )
        else:
            # Fall back to AprilTag detection.
            grounded = self._object_poses.get(BOX_TAG_ID)
            if grounded is None:
                return self._abort(goal_handle, f'No box_position provided and AprilTag {BOX_TAG_ID} not visible.')

            try:
                tf = self.tf_buffer.lookup_transform(
                    self._base_frame,
                    grounded.header.frame_id,
                    grounded.header.stamp,
                    timeout=rclpy.duration.Duration(seconds=1.0),
                )
            except Exception as e:
                return self._abort(goal_handle, f'TF lookup failed: {e}')

            cp = grounded.pose.position
            self.get_logger().info(
                f'[load_object_into_box] tag {BOX_TAG_ID} in camera ({grounded.header.frame_id}): '
                f'pos=({cp.x:.4f}, {cp.y:.4f}, {cp.z:.4f})'
            )
            box_in_base = tf2_geometry_msgs.do_transform_pose_stamped(grounded, tf)
            bp = box_in_base.pose.position
            self.get_logger().info(
                f'[load_object_into_box] tag {BOX_TAG_ID} in base_link (raw): '
                f'pos=({bp.x:.4f}, {bp.y:.4f}, {bp.z:.4f})'
            )

        current_ee_tf = self.tf_buffer.lookup_transform(
            self._base_frame, self._ee_link,
            rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0),
        )
        box_in_base.pose.orientation = current_ee_tf.transform.rotation
        box_in_base.pose.position.z += self._loading_height

        offset = _rotate_vector(box_in_base.pose.orientation, self._grasp_point_offset)
        box_in_base.pose.position.x -= offset.x
        box_in_base.pose.position.y -= offset.y
        box_in_base.pose.position.z -= offset.z

        p = box_in_base.pose.position
        self.get_logger().info(
            f'[load_object_into_box] EE target (tag {BOX_TAG_ID}) in base_link '
            f'(+{self._loading_height:.2f}m, grasp_point offset applied): '
            f'pos=({p.x:.4f}, {p.y:.4f}, {p.z:.4f})'
        )

        fb('moving_to_loading_position')
        ok, message = self._move_to_pose(box_in_base, velocity_scale=self._lower_vel_scale)
        if not ok:
            return self._abort(goal_handle, f'Move to loading position failed: {message}')

        fb('opening_gripper')
        if not self._gripper_cmd(self._gripper_open):
            return self._abort(goal_handle, 'Gripper open failed.')

        fb('done')
        goal_handle.succeed()
        r = LoadObjectIntoBox.Result()
        r.success = True
        r.message = 'Object loaded into box.'
        return r

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _resolve_pre_load_path(self) -> Path:
        if self._pre_load_path:
            return Path(self._pre_load_path).expanduser()

        share_path = Path(get_package_share_directory('manipulation')) / 'pre_load_sideways.json'
        if share_path.exists():
            return share_path

        return Path(__file__).resolve().parents[1] / 'pre_load_sideways.json'

    def _move_to_joint_positions(
        self,
        joint_positions: dict[str, float],
        timeout_sec: float = 20.0,
        velocity_scale: Optional[float] = None,
    ) -> tuple[bool, str]:
        if not self._move_group_client.wait_for_server(timeout_sec=5.0):
            return False, 'MoveGroup action server not available'

        req = MotionPlanRequest()
        req.group_name = self._move_group
        req.num_planning_attempts = 5
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = self._vel_scale if velocity_scale is None else float(velocity_scale)
        req.max_acceleration_scaling_factor = self._accel_scale

        goal_constraints = Constraints()
        for name, position in joint_positions.items():
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(position)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            goal_constraints.joint_constraints.append(jc)
        req.goal_constraints.append(goal_constraints)

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        event = threading.Event()
        result_holder = [None]

        def done_cb(future):
            result_holder[0] = future.result()
            event.set()

        future = self._move_group_client.send_goal_async(goal)
        future.add_done_callback(lambda f: f.result().get_result_async().add_done_callback(done_cb))
        event.wait(timeout=timeout_sec)

        if result_holder[0] is None:
            return False, 'MoveGroup joint goal timed out'

        code = result_holder[0].result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            return False, f'MoveGroup joint goal failed with code {code}'
        return True, 'MoveGroup joint goal succeeded'

    def _move_to_pose(
        self,
        target: PoseStamped,
        timeout_sec: float = 15.0,
        velocity_scale: Optional[float] = None,
    ) -> tuple[bool, str]:
        if not self._move_group_client.wait_for_server(timeout_sec=5.0):
            return False, 'MoveGroup action server not available'

        req = MotionPlanRequest()
        req.group_name = self._move_group
        req.num_planning_attempts = 5
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = self._vel_scale if velocity_scale is None else float(velocity_scale)
        req.max_acceleration_scaling_factor = self._accel_scale

        req.workspace_parameters = WorkspaceParameters()
        req.workspace_parameters.header.frame_id = self._base_frame
        req.workspace_parameters.min_corner = Vector3(x=-2.0, y=-2.0, z=-2.0)
        req.workspace_parameters.max_corner = Vector3(x= 2.0, y= 2.0, z= 2.0)

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.01, 0.01, 0.01]

        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = target.header.frame_id
        pos_constraint.link_name = self._ee_link
        pos_constraint.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
        pos_constraint.constraint_region = BoundingVolume()
        pos_constraint.constraint_region.primitives.append(box)
        pos_constraint.constraint_region.primitive_poses.append(target.pose)
        pos_constraint.weight = 1.0

        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = target.header.frame_id
        ori_constraint.link_name = self._ee_link
        ori_constraint.orientation = target.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.1
        ori_constraint.absolute_y_axis_tolerance = 0.1
        ori_constraint.absolute_z_axis_tolerance = 0.1
        ori_constraint.weight = 1.0

        goal_constraints = Constraints()
        goal_constraints.position_constraints.append(pos_constraint)
        goal_constraints.orientation_constraints.append(ori_constraint)
        req.goal_constraints.append(goal_constraints)

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        event = threading.Event()
        result_holder = [None]

        def done_cb(future):
            result_holder[0] = future.result()
            event.set()

        future = self._move_group_client.send_goal_async(goal)
        future.add_done_callback(lambda f: f.result().get_result_async().add_done_callback(done_cb))
        event.wait(timeout=timeout_sec)

        if result_holder[0] is None:
            return False, 'MoveGroup timed out'

        code = result_holder[0].result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            return False, f'MoveGroup failed with code {code}'
        return True, 'MoveGroup succeeded'

    def _gripper_cmd(self, position: float, max_effort: float = 50.0, timeout_sec: float = 10.0) -> bool:
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Gripper action server not available')
            return False

        goal = GripperCommand.Goal()
        goal.command.position   = position
        goal.command.max_effort = max_effort
        self.get_logger().info(f'Commanding gripper to position={position:.3f}')

        event = threading.Event()
        result_holder = [None]

        def done_cb(future):
            result_holder[0] = future.result()
            event.set()

        def goal_response_cb(future):
            gh = future.result()
            if not gh.accepted:
                self.get_logger().error('Gripper goal was rejected')
                event.set()
                return
            gh.get_result_async().add_done_callback(done_cb)

        future = self._gripper_client.send_goal_async(goal)
        future.add_done_callback(goal_response_cb)
        event.wait(timeout=timeout_sec)

        if result_holder[0] is None:
            self.get_logger().error('Gripper command timed out or was rejected')
            return False

        result = result_holder[0].result
        self.get_logger().info(
            f'Gripper result: position={result.position:.3f}, stalled={result.stalled}, reached_goal={result.reached_goal}'
        )
        return result.reached_goal or result.stalled

    def _abort(self, goal_handle, message: str):
        self.get_logger().error(message)
        goal_handle.abort()
        r = LoadObjectIntoBox.Result()
        r.success = False
        r.message = message
        return r


def _rotate_vector(q: Quaternion, v: Vector3) -> Vector3:
    x, y, z = v.x, v.y, v.z
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    r = Vector3()
    r.x = x + qw * tx + (qy * tz - qz * ty)
    r.y = y + qw * ty + (qz * tx - qx * tz)
    r.z = z + qw * tz + (qx * ty - qy * tx)
    return r


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlaceNode()
    executor = MultiThreadedExecutor(num_threads=2)
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
