import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from interfaces.msg import DeliveryState
from interfaces.srv import CheckBaseReady


class DeliveryNode(Node):
    def __init__(self) -> None:
        super().__init__('turtlebot_delivery_node')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.delivery_state_publisher = self.create_publisher(DeliveryState, '/delivery_state', 10)
        self.create_service(CheckBaseReady, '/check_base_ready', self.handle_check_base_ready)
        self.create_service(Trigger, '/start_delivery_cycle', self.handle_start_delivery_cycle)
        self.unload_timer = None
        self.current_state = 'idle'
        self.loading_pose = self._make_pose('map', 0.0, 0.0)
        self.unloading_pose = self._make_pose('map', 1.0, 0.0)
        self._publish_delivery_state(base_ready=True, carrying_load=False)
        self.get_logger().info('turtlebot_delivery_node ready')

    def handle_check_base_ready(self, request, response):
        del request
        response.ready = self.current_state in {'idle', 'at_loading'}
        response.message = f'Base state is {self.current_state}.'
        return response

    def handle_start_delivery_cycle(self, request, response):
        del request
        self.start_delivery_cycle()
        response.success = True
        response.message = 'Started stub delivery cycle.'
        return response

    def start_delivery_cycle(self) -> None:
        self.current_state = 'navigating_to_unloading'
        self._publish_delivery_state(base_ready=False, carrying_load=True)
        self._send_nav_goal(self.unloading_pose)

    def _send_nav_goal(self, pose: PoseStamped) -> None:
        if not self.nav_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warning('NavigateToPose action server not available; simulating success.')
            self._nav_result_callback(None)
            return
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        send_future = self.nav_action_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warning('Navigation goal was rejected; simulating completion.')
            self._nav_result_callback(None)
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._nav_result_callback)

    def _nav_result_callback(self, future) -> None:
        del future
        if self.current_state == 'navigating_to_unloading':
            self.current_state = 'at_unloading'
            self._publish_delivery_state(base_ready=False, carrying_load=True)
            self._start_unload_timer()
        elif self.current_state == 'navigating_to_loading':
            self.current_state = 'at_loading'
            self._publish_delivery_state(base_ready=True, carrying_load=False)

    def _start_unload_timer(self) -> None:
        self.current_state = 'waiting_for_unload'
        self._publish_delivery_state(base_ready=False, carrying_load=True)
        self.unload_timer = self.create_timer(10.0, self._on_unload_complete)

    def _on_unload_complete(self) -> None:
        if self.unload_timer is not None:
            self.unload_timer.cancel()
            self.unload_timer = None
        self.current_state = 'navigating_to_loading'
        self._publish_delivery_state(base_ready=False, carrying_load=False)
        self._send_nav_goal(self.loading_pose)

    def _publish_delivery_state(self, base_ready: bool, carrying_load: bool) -> None:
        msg = DeliveryState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.state = self.current_state
        msg.base_ready = base_ready
        msg.carrying_load = carrying_load
        self.delivery_state_publisher.publish(msg)

    def _make_pose(self, frame_id: str, x: float, y: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0
        return pose


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DeliveryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
