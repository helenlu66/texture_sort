"""
Standalone test: calls /deliver with target='unloading' exactly as task_manager does.

Usage:
    python3 src/task_manager/task_manager/test_deliver_action.py
or after sourcing the workspace:
    ros2 run task_manager test_deliver_action
"""

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from interfaces.action import Delivery

SERVICE_TIMEOUT = 10.0   # matches task_manager default service_timeout_sec
ACTION_TIMEOUT  = 90.0   # matches task_manager default action_timeout_sec


class DeliveryTester(Node):
    def __init__(self) -> None:
        super().__init__('deliver_tester')
        self._client = ActionClient(self, Delivery, '/deliver')

    def run(self) -> bool:
        self.get_logger().info('Waiting for /deliver action server...')
        if not self._client.wait_for_server(timeout_sec=SERVICE_TIMEOUT):
            self.get_logger().error('/deliver server unavailable')
            return False

        goal = Delivery.Goal()
        goal.target = 'unloading'
        self.get_logger().info(f'Sending goal: target={goal.target!r}')

        gf = self._client.send_goal_async(goal, feedback_callback=self._feedback_cb)
        rclpy.spin_until_future_complete(self, gf, timeout_sec=SERVICE_TIMEOUT)
        gh = gf.result()
        if gh is None or not gh.accepted:
            self.get_logger().error('Goal rejected')
            return False

        self.get_logger().info('Goal accepted — waiting for result...')
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf, timeout_sec=ACTION_TIMEOUT)
        r = rf.result()
        if r is None:
            self.get_logger().error('/deliver timed out')
            return False

        result = r.result
        success = result is not None and bool(getattr(result, 'success', False))
        message = str(getattr(result, 'message', '')) if result is not None else ''
        if success:
            self.get_logger().info(f'SUCCESS: {message}')
        else:
            self.get_logger().error(f'FAILED: {message}')
        return success

    def _feedback_cb(self, feedback_msg) -> None:
        self.get_logger().info(f'[feedback] state={feedback_msg.feedback.state!r}')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DeliveryTester()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        success = node.run()
    except KeyboardInterrupt:
        success = False
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(0 if success else 1)


if __name__ == '__main__':
    main()
