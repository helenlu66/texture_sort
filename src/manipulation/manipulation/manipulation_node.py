from typing import Optional
import threading

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.action import ActionServer, ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import (
    PoseStamped, Vector3,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints,
    PositionConstraint, OrientationConstraint,
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
GRIPPER_CLOSED = 0.6   # adjust for object thickness

APPROACH_HEIGHT = 0.12   # metres above object before descending
GRASP_Z_OFFSET  = 0.01   # metres above tag centre at grasp
LIFT_HEIGHT     = 0.15   # metres to lift after closing gripper
SAFE_GRASP_Z_OFFSET = 0.10  # metres above detected object for early testing


class ManipulationNode(Node):
    def __init__(self) -> None:
        super().__init__('manipulation_node')

        self.declare_parameter('groundings_topic', '/groundings')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('ee_link', 'end_effector_link')
        self.declare_parameter('move_group', 'manipulator')
        self.declare_parameter('vel_scale', 0.3)
        self.declare_parameter('accel_scale', 0.3)

        groundings_topic = str(self.get_parameter('groundings_topic').value)
        self._base_frame  = str(self.get_parameter('base_frame').value)
        self._ee_link     = str(self.get_parameter('ee_link').value)
        self._move_group  = str(self.get_parameter('move_group').value)
        self._vel_scale   = float(self.get_parameter('vel_scale').value)
        self._accel_scale = float(self.get_parameter('accel_scale').value)
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

        self.execute_grasp_action_server = ActionServer(
            self,
            ExecuteGrasp,
            '/execute_grasp',
            self.execute_grasp_callback,
            callback_group=self._action_callback_group,
        )

        self.get_logger().info('manipulation_node ready')

    # ------------------------------------------------------------------ #
    # Subscriptions                                                        #
    # ------------------------------------------------------------------ #

    def _groundings_callback(self, msg: ObjectGroundingArray) -> None:
        for g in msg.objects:
            self._object_poses[g.object_id] = g.pose

    def _joint_state_callback(self, msg: JointState) -> None:
        self.current_joint_state = msg

    # ------------------------------------------------------------------ #
    # ExecuteGrasp action                                                  #
    # ------------------------------------------------------------------ #

    def execute_grasp_callback(self, goal_handle):
        oid = goal_handle.request.object_id

        def fb(state: str):
            msg = ExecuteGrasp.Feedback()
            msg.state = state
            goal_handle.publish_feedback(msg)
            self.get_logger().info(f'[grasp {oid}] {state}')

        fb('resolving_target_pose')
        try:
            target_in_base = self._resolve_grasp_target(oid)
        except Exception as e:
            return self._abort(goal_handle, f'TF lookup failed: {e}')
        if target_in_base is None:
            return self._abort(
                goal_handle,
                f'No grounding for object {oid}.',
            )

        moved, message = self._move_to_pose(target_in_base)
        if not moved:
            return self._abort(goal_handle, f'Grasp motion failed: {message}')

        fb('done')
        goal_handle.succeed()
        r = ExecuteGrasp.Result()
        r.success = True
        r.message = f'Moved {self._ee_link} to target pose for object {oid}.'
        return r

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _resolve_grasp_target(self, object_id: int) -> Optional[PoseStamped]:
        grounded_pose = self._object_poses.get(object_id)
        if grounded_pose is None:
            return None

        if grounded_pose.header.frame_id == self._base_frame:
            target = PoseStamped()
            target.header = grounded_pose.header
            target.pose = grounded_pose.pose
        else:
            tf = self.tf_buffer.lookup_transform(
                self._base_frame,
                grounded_pose.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
            target = tf2_geometry_msgs.do_transform_pose_stamped(grounded_pose, tf)

        target.pose.position.z += SAFE_GRASP_Z_OFFSET
        return target

    def _move_to_pose(self, target: PoseStamped, timeout_sec: float = 15.0) -> tuple[bool, str]:
        if not self._move_group_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('MoveGroup action server not available')
            return False, 'MoveGroup action server not available'

        req = MotionPlanRequest()
        req.group_name = self._move_group
        req.num_planning_attempts = 5
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = self._vel_scale
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

        event = threading.Event()
        result_holder = [None]

        def done_cb(future):
            result_holder[0] = future.result()
            event.set()

        future = self._gripper_client.send_goal_async(goal)
        future.add_done_callback(lambda f: f.result().get_result_async().add_done_callback(done_cb))
        event.wait(timeout=timeout_sec)

        return result_holder[0] is not None

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


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ManipulationNode()
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
