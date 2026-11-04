"""
ROS 2 (rclpy) wrapper: runs the behaviour tree at 10 Hz and publishes [target_speed, target_lane].
STATUS: not executed in this project (no ROS 2 installation was used); interface skeleton only.
Subscribes: /ego/odom, /lead/odom, /oncoming/odom (nav_msgs/Odometry, route-aligned x = station [m]).
Publishes : /bt/cmd (std_msgs/Float32MultiArray) -> [speed_mps, lane(0|1)]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import behavior_tree as bt

try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float32MultiArray
except ImportError:
    rclpy = None


class BTNode(Node if rclpy else object):
    def __init__(self):
        super().__init__("behavior_tree")
        self.tree = bt.build_tree(); self.w = bt.World(0)
        self.w.cv_times = np.array([])
        self.create_subscription(Odometry, "/ego/odom", self.on_ego, 10)
        self.create_subscription(Odometry, "/lead/odom", self.on_lead, 10)
        self.create_subscription(Odometry, "/oncoming/odom", self.on_onc, 10)
        self.pub = self.create_publisher(Float32MultiArray, "/bt/cmd", 10)
        self.create_timer(0.1, self.tick)

    def on_ego(self, m):  self.w.s, self.w.v, self.w.lat = m.pose.pose.position.x, m.twist.twist.linear.x, float(np.clip(m.pose.pose.position.y / 3.5, 0, 1))
    def on_lead(self, m): self.w.lead_s, self.w.lead_v = m.pose.pose.position.x, m.twist.twist.linear.x
    def on_onc(self, m):  self.w.onc_s, self.w.onc_v = m.pose.pose.position.x, -abs(m.twist.twist.linear.x)

    def tick(self):
        self.w.t = self.get_clock().now().nanoseconds * 1e-9
        bb = bt.bb_from_world(self.w); bb["cmd"] = dict(speed=0.0, lane=0); self.tree.tick(bb)
        msg = Float32MultiArray(); msg.data = [float(bb["cmd"]["speed"]), float(bb["cmd"]["lane"])]; self.pub.publish(msg)


def main():
    if rclpy is None:
        raise SystemExit("rclpy not available: source your ROS 2 environment first")
    rclpy.init(); rclpy.spin(BTNode()); rclpy.shutdown()


if __name__ == "__main__":
    main()
