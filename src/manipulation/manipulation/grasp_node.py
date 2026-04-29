from typing import Optional
import copy
import json
from pathlib import Path
import threading
import time

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
from std_msgs.msg import Bool
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

from interfaces.action import ExecuteGrasp
from interfaces.msg import ObjectGroundingArray


# Robotiq 2F-85: 0.0 = fully open, 0.8 = fully closed
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.79   # Robotiq 2F-85 closed position is about 0.7929

APPROACH_HEIGHT = 0.12   # metres above object before descending
DEFAULT_GRASP_Z_OFFSET = 0.0 # tuned meters above tag center


class GraspNode(Node):
    def __init__(self) -> None:
        super().__init__('grasp_node')

        self.declare_parameter('groundings_topic', '/groundings')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('ee_link', 'end_effector_link')
        self.declare_parameter('move_group', 'manipulator')
        self.declare_parameter('vel_scale', 0.3)
        self.declare_parameter('lower_vel_scale', 0.1)
        self.declare_parameter('accel_scale', 0.3)
        self.declare_parameter('grasp_z_offset', DEFAULT_GRASP_Z_OFFSET)
        self.declare_parameter('approach_height', APPROACH_HEIGHT)
        self.declare_parameter('gripper_closed_position', GRIPPER_CLOSED)
        self.declare_parameter('gripper_open_position', GRIPPER_OPEN)
        self.declare_parameter('grasp_point_offset_x', 0.0)
        self.declare_parameter('grasp_point_offset_y', 0.0)
        self.declare_parameter('grasp_point_offset_z', 0.13)
        self.declare_parameter('pre_grasp_path', '')

        groundings_topic = str(self.get_parameter('groundings_topic').value)
        self._base_frame  = str(self.get_parameter('base_frame').value)
        self._ee_link     = str(self.get_parameter('ee_link').value)
        self._move_group  = str(self.get_parameter('move_group').value)
        self._vel_scale   = float(self.get_parameter('vel_scale').value)
        self._lower_vel_scale = float(self.get_parameter('lower_vel_scale').value)
        self._accel_scale = float(self.get_parameter('accel_scale').value)
        self._grasp_z_offset = float(self.get_parameter('grasp_z_offset').value)
        self._approach_height = float(self.get_parameter('approach_height').value)
        self._gripper_closed_position = float(self.get_parameter('gripper_closed_position').value)
        self._gripper_open_position = float(self.get_parameter('gripper_open_position').value)
        self._grasp_point_offset = Vector3(
            x=float(self.get_parameter('grasp_point_offset_x').value),
            y=float(self.get_parameter('grasp_point_offset_y').value),
            z=float(self.get_parameter('grasp_point_offset_z').value),
        )
        self._pre_grasp_path = str(self.get_parameter('pre_grasp_path').value)
        self._action_callback_group = ReentrantCallbackGroup()

        self.current_joint_state: Optional[JointState] = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._object_poses: dict[int, PoseStamped] = {}
        self._grasp_state = False

        self.grasp_state_publisher = self.create_publisher(Bool, '/grasp_state', 10)
        self.create_subscription(JointState, '/joint_states', self._joint_state_callback, 10)
        self.create_subscription(ObjectGroundingArray, groundings_topic, self._groundings_callback, 10)

        self._move_group_client = ActionClient(
            self,
            MoveGroup,
            '/move_action',
            callback_group=self._action_callback_group,
        )
        self._gripper_client = ActionClient(
            self,
            GripperCommand,
            '/robotiq_gripper_controller/gripper_cmd',
            callback_group=self._action_callback_group,
        )

        self.grasp_then_reset_action_server = ActionServer(
            self,
            ExecuteGrasp,
            '/grasp_then_reset',
            self.grasp_then_reset_callback, # same sequence as pickup but without picking up the object
            callback_group=self._action_callback_group,
        )

        self.reset_action_server = ActionServer(
            self,
            ExecuteGrasp,
            '/reset_grasp',
            self.grasp_then_reset_callback,  # same sequence as grasp_then_reset but with open_back_up=True and lift=False
            callback_group=self._action_callback_group,
        )
        
        self.execute_grasp_action_server = ActionServer(
            self,
            ExecuteGrasp,
            '/execute_grasp',
            self.execute_grasp_callback,
            callback_group=self._action_callback_group,
        )
        self.pickup_action_server = ActionServer(
            self,
            ExecuteGrasp,
            '/pickup',
            self.pickup_callback,
            callback_group=self._action_callback_group,
        )

        self.get_logger().info('grasp_node ready')

    # ------------------------------------------------------------------ #
    # Subscriptions                                                        #
    # ------------------------------------------------------------------ #

    def _groundings_callback(self, msg: ObjectGroundingArray) -> None:
        for g in msg.objects:
            self._object_poses[g.object_id] = g.pose

    def _joint_state_callback(self, msg: JointState) -> None:
        self.current_joint_state = msg

    # ------------------------------------------------------------------ #
    # Actions                                                             #
    # ------------------------------------------------------------------ #

    def grasp_then_reset_callback(self, goal_handle):
        """Grasp the object, then opens the gripper, returs to the approach height, and returns to the pre-grasp pose without picking up the object. Used for collecting tactile data from each object."""
        success, message = self._run_grasp_sequence(goal_handle, open_back_up=True, lift=True)
        if not success:
            return self._abort(goal_handle, message)
        
        goal_handle.succeed()
        r = ExecuteGrasp.Result()
        r.success = True
        r.message = message
        return r

    def reset_grasp_callback(self, goal_handle):
        """Opens the gripper, returs to the approach height, and returns to the pre-grasp pose"""
        # open the gripper
        if not self._gripper_cmd(self._gripper_open_position):
            return self._abort(goal_handle, 'Gripper open command failed.')
        
        # go back up to the approach height. It's the current gripper pose with z = approach height
        current_ee_tf = self.tf_buffer.lookup_transform(
            self._base_frame,
            self._ee_link,
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=1.0),
        )
        approach_target = PoseStamped()
        approach_target.header = current_ee_tf.header
        approach_target.pose = current_ee_tf.transform
        approach_target.pose.position.z = self._approach_height
        moved, message = self._move_to_pose(approach_target)
        if not moved:
            return self._abort(goal_handle, f'Approach motion failed: {message}')
        
        # return to pre-grasp pose
        returned, message = self.return_to_pre_grasp()
        if not returned:
            return self._abort(goal_handle, f'Failed to return to pre-grasp pose: {message}')
        goal_handle.succeed()
        r = ExecuteGrasp.Result()
        r.success = True
        r.message = 'Grasp reset successful.'
        return r


    def execute_grasp_callback(self, goal_handle):
        success, message = self._run_grasp_sequence(goal_handle, lift=False)
        if not success:
            return self._abort(goal_handle, message)

        goal_handle.succeed()
        r = ExecuteGrasp.Result()
        r.success = True
        r.message = message
        return r

    def pickup_callback(self, goal_handle):
        success, message = self._run_grasp_sequence(goal_handle, lift=True)
        if not success:
            return self._abort(goal_handle, message)

        goal_handle.succeed()
        r = ExecuteGrasp.Result()
        r.success = True
        r.message = message
        return r

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    def _run_grasp_sequence(self, goal_handle, open_back_up:bool=False, lift: bool=False) -> tuple[bool, str]:
        """Grasp the object, with the option to open the gripper back up and the option to go back up.
        Args:
            goal_handle: The goal handle for the grasp action.
            open_back_up: Whether to open the gripper back up after grasping.
            lift: Whether to lift the object after grasping.
        Returns:
            A tuple of (success, message).
        """

        oid = goal_handle.request.object_id

        def fb(state: str):
            msg = ExecuteGrasp.Feedback()
            msg.state = state
            goal_handle.publish_feedback(msg)
            self.get_logger().info(f'[grasp {oid}] {state}')

        fb('resolving_target_pose')
        try:
            grasp_target_in_base = self._resolve_grasp_target(oid)
        except Exception as e:
            return False, f'TF lookup failed: {e}'
        if grasp_target_in_base is None:
            return False, f'No grounding for object {oid}.'

        p = grasp_target_in_base.pose.position
        q = grasp_target_in_base.pose.orientation
        # print out the current ee's pose in base_link for debugging
        current_ee_tf = self.tf_buffer.lookup_transform(
            self._base_frame,
            self._ee_link,
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=1.0),
        )
        current_ee_pos = current_ee_tf.transform.translation
        current_ee_ori = current_ee_tf.transform.rotation
        self.get_logger().info(
            f'[grasp {oid}] current {self._ee_link} pose in {self._base_frame}: '
            f'pos=({current_ee_pos.x:.4f}, {current_ee_pos.y:.4f}, {current_ee_pos.z:.4f})  '
            f'quat=({current_ee_ori.x:.4f}, {current_ee_ori.y:.4f}, {current_ee_ori.z:.4f}, {current_ee_ori.w:.4f})'
        )
        self.get_logger().info(
            f'[grasp {oid}] grasp_point target in base_link: '
            f'pos=({p.x:.4f}, {p.y:.4f}, {p.z:.4f})  '
            f'quat=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})'
        )

        approach_grasp_target = copy.deepcopy(grasp_target_in_base)
        approach_grasp_target.pose.position.z += self._approach_height
        ap = approach_grasp_target.pose.position
        self.get_logger().info(
            f'[grasp {oid}] grasp_point approach target in base_link: '
            f'pos=({ap.x:.4f}, {ap.y:.4f}, {ap.z:.4f})'
        )

        approach_ee_target = self._ee_target_for_grasp_point(approach_grasp_target)
        grasp_ee_target = self._ee_target_for_grasp_point(grasp_target_in_base)

        fb('moving_to_approach')
        moved, message = self._move_to_pose(approach_ee_target)
        if not moved:
            return False, f'Approach motion failed: {message}'
        
        # wait for 2 seconds
        # time.sleep(2.0)
        # # resolve target again for better x, y precision after moving to approach pose
        # grasp_target_in_base = self._resolve_grasp_target(oid)
        # if grasp_target_in_base is None:
        #     return False, f'No grounding for object {oid} after moving to approach pose.'
        # # use the old z but updated x, y for the grasp pose to try to correct for any x, y error in the original grounding
        # new_grasp_ee_target = self._ee_target_for_grasp_point(grasp_target_in_base)
        # new_grasp_ee_target.pose.position.z = grasp_ee_target.pose.position.z

        fb('lowering_to_object')
        moved, message = self._move_to_pose(
            grasp_ee_target,
            velocity_scale=self._lower_vel_scale,
        )
        if not moved:
            return False, f'Descent motion failed: {message}'

        fb('closing_gripper')
        if not self._gripper_cmd(self._gripper_closed_position):
            return False, 'Gripper close command failed.'
        self._publish_grasp_state(True)

        if open_back_up:
            fb('opening_gripper')
            if not self._gripper_cmd(self._gripper_open_position):
                return False, 'Gripper open command failed after grasping.'
            self._publish_grasp_state(False)

        if lift:
            # lift_target = copy.deepcopy(grasp_ee_target)
            # lift_target.pose.position.z += self._approach_height
            # p = lift_target.pose.position
            # self.get_logger().info(
            #     f'[grasp {oid}] lift target for {self._ee_link} in base_link: '
            #     f'pos=({p.x:.4f}, {p.y:.4f}, {p.z:.4f})'
            # )

            fb('lifting')
            moved, message = self.return_to_approach_height(grasp_ee_target)
            if not moved:
                return False, f'Lift motion failed: {message}'

            fb('returning_to_pre_grasp')
            returned, message = self.return_to_pre_grasp()
            if not returned:
                return False, f'Return to pre_grasp failed: {message}'

        fb('done')
        if lift:
            return True, f'Grasped object {oid} and returned to pre_grasp.'
        return True, f'Moved to object {oid} and closed gripper.'


    def _resolve_grasp_target(self, object_id: int) -> Optional[PoseStamped]:
        grounded_pose = self._object_poses.get(object_id)
        if grounded_pose is None:
            return None

        p = grounded_pose.pose.position
        q = grounded_pose.pose.orientation
        self.get_logger().info(
            f'[debug] object in camera ({grounded_pose.header.frame_id}): '
            f'pos=({p.x:.4f}, {p.y:.4f}, {p.z:.4f})  '
            f'quat=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})'
        )

        tf = self.tf_buffer.lookup_transform(
            self._base_frame,
            grounded_pose.header.frame_id,
            grounded_pose.header.stamp,
            timeout=rclpy.duration.Duration(seconds=1.0),
        )
        t = tf.transform.translation
        r = tf.transform.rotation
        self.get_logger().info(
            f'[debug] tf {grounded_pose.header.frame_id} -> {self._base_frame}: '
            f'trans=({t.x:.4f}, {t.y:.4f}, {t.z:.4f})  '
            f'quat=({r.x:.4f}, {r.y:.4f}, {r.z:.4f}, {r.w:.4f})'
        )
        target = tf2_geometry_msgs.do_transform_pose_stamped(grounded_pose, tf)

        p = target.pose.position
        q = target.pose.orientation
        self.get_logger().info(
            f'[debug] object in base_link (pre-offset): '
            f'pos=({p.x:.4f}, {p.y:.4f}, {p.z:.4f})  '
            f'quat=({q.x:.4f}, {q.y:.4f}, {q.z:.4f}, {q.w:.4f})'
        )

        target.pose.position.z += self._grasp_z_offset

        current_ee_tf = self.tf_buffer.lookup_transform(
            self._base_frame,
            self._ee_link,
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=1.0),
        )
        target.pose.orientation = current_ee_tf.transform.rotation

        # self.get_logger().info(
        #     f'[debug] grasp orientation in base_link: '
        #     f'using current {self._ee_link} orientation and ignoring tag orientation'
        # )
        return target

    def _ee_target_for_grasp_point(self, grasp_point_target: PoseStamped) -> PoseStamped:
        ee_target = copy.deepcopy(grasp_point_target)
        offset = _rotate_vector(ee_target.pose.orientation, self._grasp_point_offset)
        ee_target.pose.position.x -= offset.x
        ee_target.pose.position.y -= offset.y
        ee_target.pose.position.z -= offset.z

        p = ee_target.pose.position
        self.get_logger().info(
            f'[debug] commanded {self._ee_link} target after grasp_point offset: '
            f'pos=({p.x:.4f}, {p.y:.4f}, {p.z:.4f})  '
            f'offset_in_base=({offset.x:.4f}, {offset.y:.4f}, {offset.z:.4f})'
        )
        return ee_target

    def return_to_approach_height(self, target: PoseStamped) -> tuple[bool, str]:
        """Move the end effector back up to the approach height above the target, without changing x/y or orientation. Used for going back up after grasping."""
        approach_target = copy.deepcopy(target)
        approach_target.pose.position.z += self._approach_height
        return self._move_to_pose(approach_target)
    
    def return_to_pre_grasp(self) -> tuple[bool, str]:
        pre_grasp_path = self._resolve_pre_grasp_path()
        with pre_grasp_path.open('r', encoding='utf-8') as f:
            pre_grasp = json.load(f)

        arm_joints = pre_grasp.get('arm_joints')
        if not isinstance(arm_joints, dict) or not arm_joints:
            return False, f'No arm_joints found in {pre_grasp_path}'

        self.get_logger().info(
            f'Returning to pre_grasp from {pre_grasp_path}: '
            + ', '.join(f'{name}={position:.4f}' for name, position in arm_joints.items())
        )
        return self._move_to_joint_positions(arm_joints)

    def _resolve_pre_grasp_path(self) -> Path:
        if self._pre_grasp_path:
            return Path(self._pre_grasp_path).expanduser()

        share_path = Path(get_package_share_directory('manipulation')) / 'pre_grasp.json'
        if share_path.exists():
            return share_path

        return Path(__file__).resolve().parents[1] / 'pre_grasp.json'

    def _move_to_joint_positions(
        self,
        joint_positions: dict[str, float],
        timeout_sec: float = 20.0,
    ) -> tuple[bool, str]:
        if not self._move_group_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveGroup action server not available')
            return False, 'MoveGroup action server not available'

        req = MotionPlanRequest()
        req.group_name = self._move_group
        req.num_planning_attempts = 5
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = self._vel_scale
        req.max_acceleration_scaling_factor = self._accel_scale

        goal_constraints = Constraints()
        for name, position in joint_positions.items():
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = name
            joint_constraint.position = float(position)
            joint_constraint.tolerance_above = 0.01
            joint_constraint.tolerance_below = 0.01
            joint_constraint.weight = 1.0
            goal_constraints.joint_constraints.append(joint_constraint)
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
            self.get_logger().error('MoveGroup joint goal timed out')
            return False, 'MoveGroup joint goal timed out'

        code = result_holder[0].result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            message = f'MoveGroup joint goal failed with code {code}'
            self.get_logger().error(message)
            return False, message
        return True, 'MoveGroup joint goal succeeded'

    def _move_to_pose(
        self,
        target: PoseStamped,
        timeout_sec: float = 15.0,
        velocity_scale: Optional[float] = None,
    ) -> tuple[bool, str]:
        if not self._move_group_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveGroup action server not available')
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

        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = target.header.frame_id
        pos_constraint.link_name = self._ee_link
        pos_constraint.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.01, 0.01, 0.01]
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
            self.get_logger().error('MoveGroup timed out')
            return False, 'MoveGroup timed out'

        code = result_holder[0].result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            message = f'MoveGroup failed with code {code}'
            self.get_logger().error(message)
            return False, message
        return True, 'MoveGroup succeeded'

    def _gripper_cmd(self, position: float, max_effort: float = 50.0, timeout_sec: float = 10.0) -> bool:
        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Gripper action server not available')
            return False

        goal = GripperCommand.Goal()
        goal.command.position   = position
        goal.command.max_effort = max_effort
        self.get_logger().info(
            f'Commanding gripper to position={position:.3f}, max_effort={max_effort:.1f}'
        )

        event = threading.Event()
        result_holder = [None]

        def done_cb(future):
            result_holder[0] = future.result()
            event.set()

        def goal_response_cb(future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error('Gripper goal was rejected')
                event.set()
                return
            goal_handle.get_result_async().add_done_callback(done_cb)

        future = self._gripper_client.send_goal_async(goal)
        future.add_done_callback(goal_response_cb)
        event.wait(timeout=timeout_sec)

        if result_holder[0] is None:
            self.get_logger().error('Gripper command timed out or was rejected')
            return False

        result = result_holder[0].result
        self.get_logger().info(
            f'Gripper result: position={result.position:.3f}, '
            f'effort={result.effort:.3f}, stalled={result.stalled}, reached_goal={result.reached_goal}'
        )
        return result.reached_goal or result.stalled

    def _abort(self, goal_handle, message: str):
        self.get_logger().error(message)
        goal_handle.abort()
        r = ExecuteGrasp.Result()
        r.success = False
        r.message = message
        return r

    def _publish_grasp_state(self, grasped: bool) -> None:
        msg = Bool()
        msg.data = grasped
        self.grasp_state_publisher.publish(msg)


def _rotate_vector(q: Quaternion, v: Vector3) -> Vector3:
    # q * v * q^-1, expanded for a unit quaternion.
    x = v.x
    y = v.y
    z = v.z
    qx = q.x
    qy = q.y
    qz = q.z
    qw = q.w

    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)

    rotated = Vector3()
    rotated.x = x + qw * tx + (qy * tz - qz * ty)
    rotated.y = y + qw * ty + (qz * tx - qx * tz)
    rotated.z = z + qw * tz + (qx * ty - qy * tx)
    return rotated


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspNode()
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
