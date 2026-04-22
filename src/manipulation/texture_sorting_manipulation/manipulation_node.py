from typing import Optional

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener
from texture_sorting_interfaces.action import ExecuteGrasp, ExecutePlace, LoadObjectIntoBox


class ManipulationNode(Node):
    def __init__(self) -> None:
        super().__init__('manipulation_node')
        self.current_joint_state: Optional[JointState] = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.grasp_state_publisher = self.create_publisher(Bool, '/grasp_state', 10)
        self.create_subscription(JointState, '/joint_states', self._joint_state_callback, 10)
        self.execute_grasp_action_server = ActionServer(self, ExecuteGrasp, '/execute_grasp', self.execute_grasp_callback)
        self.execute_place_action_server = ActionServer(self, ExecutePlace, '/execute_place', self.execute_place_callback)
        self.load_object_into_box_action_server = ActionServer(self, LoadObjectIntoBox, '/load_object_into_box', self.load_object_into_box_callback)
        self.get_logger().info('manipulation_node ready')

    def execute_grasp_callback(self, goal_handle):
        feedback = ExecuteGrasp.Feedback()
        feedback.state = 'closing_gripper'
        goal_handle.publish_feedback(feedback)
        self._publish_grasp_state(True)
        goal_handle.succeed()
        result = ExecuteGrasp.Result()
        result.success = True
        result.message = f'Stub grasp succeeded for object {goal_handle.request.object_id}.'
        return result

    def execute_place_callback(self, goal_handle):
        feedback = ExecutePlace.Feedback()
        feedback.state = 'moving_to_place_pose'
        goal_handle.publish_feedback(feedback)
        self._publish_grasp_state(False)
        goal_handle.succeed()
        result = ExecutePlace.Result()
        result.success = True
        result.message = f'Stub place succeeded for object {goal_handle.request.object_id}.'
        return result

    def load_object_into_box_callback(self, goal_handle):
        feedback = LoadObjectIntoBox.Feedback()
        feedback.state = 'loading_box'
        goal_handle.publish_feedback(feedback)
        self._publish_grasp_state(False)
        goal_handle.succeed()
        result = LoadObjectIntoBox.Result()
        result.success = True
        result.message = f'Stub load-into-box succeeded for object {goal_handle.request.object_id} in category {goal_handle.request.category_id}.'
        return result

    def _joint_state_callback(self, msg: JointState) -> None:
        self.current_joint_state = msg

    def _publish_grasp_state(self, grasped: bool) -> None:
        msg = Bool()
        msg.data = grasped
        self.grasp_state_publisher.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ManipulationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
