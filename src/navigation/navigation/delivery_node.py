import time

import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from interfaces.action import Deliver, Return
from interfaces.msg import DeliveryState
from interfaces.srv import CheckBaseReady


class DeliveryNode(Node):
    def __init__(self) -> None:
        super().__init__('delivery_node')
        self._callback_group = ReentrantCallbackGroup()
        self.current_state = 'idle'

        self.delivery_state_publisher = self.create_publisher(DeliveryState, '/delivery_state', 10)
        self.create_service(
            CheckBaseReady,
            '/check_base_ready',
            self.handle_check_base_ready,
            callback_group=self._callback_group,
        )
        self.create_service(
            Trigger,
            '/start_delivery_cycle',
            self.handle_start_delivery_cycle,
            callback_group=self._callback_group,
        )
        ActionServer(
            self,
            Deliver,
            '/deliver',
            self._deliver_callback,
            callback_group=self._callback_group,
        )
        ActionServer(
            self,
            Return,
            '/return',
            self._return_callback,
            callback_group=self._callback_group,
        )

        self._publish_delivery_state(base_ready=True, carrying_load=False)
        self.get_logger().info('delivery_node ready')

    def handle_check_base_ready(self, request, response):
        del request
        response.ready = self.current_state in {'idle', 'at_loading'}
        response.message = f'Base state is {self.current_state}.'
        return response

    def handle_start_delivery_cycle(self, request, response):
        del request
        self.current_state = 'legacy_delivery_requested'
        self._publish_delivery_state(base_ready=False, carrying_load=True)
        time.sleep(1.0)
        self.current_state = 'at_unloading'
        self._publish_delivery_state(base_ready=False, carrying_load=True)
        response.success = True
        response.message = 'Completed stub delivery cycle.'
        return response

    def _deliver_callback(self, goal_handle):
        del goal_handle.request

        def fb(state: str) -> None:
            msg = Deliver.Feedback()
            msg.state = state
            goal_handle.publish_feedback(msg)
            self.get_logger().info(f'[deliver] {state}')

        fb('delivering')
        self.current_state = 'delivering'
        self._publish_delivery_state(base_ready=False, carrying_load=True)
        time.sleep(5.0)

        fb('delivered')
        self.current_state = 'delivered'
        self._publish_delivery_state(base_ready=False, carrying_load=False)
        goal_handle.succeed()

        result = Deliver.Result()
        result.success = True
        result.message = 'Delivery completed.'
        return result

    def _return_callback(self, goal_handle):
        del goal_handle.request

        def fb(state: str) -> None:
            msg = Return.Feedback()
            msg.state = state
            goal_handle.publish_feedback(msg)
            self.get_logger().info(f'[return] {state}')

        fb('returning')
        self.current_state = 'returning'
        self._publish_delivery_state(base_ready=False, carrying_load=False)
        time.sleep(5.0)

        fb('at_loading')
        self.current_state = 'at_loading'
        self._publish_delivery_state(base_ready=True, carrying_load=False)
        goal_handle.succeed()

        result = Return.Result()
        result.success = True
        result.message = 'Returned to loading area.'
        return result

    def _publish_delivery_state(self, base_ready: bool, carrying_load: bool) -> None:
        msg = DeliveryState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.state = self.current_state
        msg.base_ready = base_ready
        msg.carrying_load = carrying_load
        self.delivery_state_publisher.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DeliveryNode()
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
