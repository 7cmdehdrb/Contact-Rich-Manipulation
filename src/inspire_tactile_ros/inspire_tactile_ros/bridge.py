"""Local UDP receiver -> ROS 2 topics. Never imports Isaac Sim."""

import json
import socket

import rclpy
from rclpy.node import Node
from std_msgs.msg import MultiArrayDimension, String, UInt8MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from .protocol import decode


def make_markers(packet, stamp):
    result = MarkerArray()
    for index, (name, position, active) in enumerate(zip(
            packet["names"], packet["positions_w_m"], packet["active"])):
        marker = Marker()
        marker.header.frame_id = packet["frame_id"]
        marker.header.stamp = stamp
        marker.ns = f"env_{packet['env_id']}/sensors"
        marker.id = index
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, position)
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.009
        marker.color.r, marker.color.g, marker.color.b = (0.1, 0.9, 0.3) if active else (0.45, 0.45, 0.45)
        marker.color.a = 1.0
        # Do not leave stale green points in RViz after a disconnect.
        marker.lifetime.sec = 1
        result.markers.append(marker)
        label = Marker()
        label.header = marker.header
        label.ns = f"env_{packet['env_id']}/labels"
        label.id = index
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x, label.pose.position.y, label.pose.position.z = map(float, position)
        label.pose.position.z += 0.012
        label.pose.orientation.w = 1.0
        label.scale.z = 0.007
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = f"{index}: {name.removeprefix('inspire_').replace('_force_sensor', '')}"
        label.lifetime.sec = 1
        result.markers.append(label)
    return result


class TactileBridge(Node):
    def __init__(self):
        super().__init__("inspire_tactile_bridge")
        port = self.declare_parameter("port", 9870).value
        topic = self.declare_parameter("topic", "/inspire/tactile").value.rstrip("/")
        self.state_pub = self.create_publisher(String, topic, 10)
        self.bits_pub = self.create_publisher(UInt8MultiArray, topic + "/binary", 10)
        self.marker_pub = self.create_publisher(MarkerArray, topic + "/markers", 10)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.socket.bind(("127.0.0.1", port))
        except BaseException:
            self.socket.close()
            raise
        self.socket.setblocking(False)
        self.last = {}
        self.warned = False
        self.create_timer(0.01, self.receive)
        self.get_logger().info(f"Listening on 127.0.0.1:{port}; publishing {topic}, /binary, /markers")

    def receive(self):
        latest = {}
        # Bounded work per callback; newest complete sample per environment wins.
        for _ in range(256):
            try:
                data, _ = self.socket.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                packet = decode(data)
            except (ValueError, TypeError, KeyError) as exc:
                if not self.warned:
                    self.get_logger().warning(f"Ignoring malformed tactile sample: {exc}")
                    self.warned = True
                continue
            env_id = packet["env_id"]
            last = self.last.get(env_id)
            if last and last[0] == packet["session"] and packet["sequence"] <= last[1]:
                continue
            self.last[env_id] = (packet["session"], packet["sequence"])
            latest[env_id] = packet
        for packet in latest.values():
            stamp = self.get_clock().now().to_msg()
            # ROS header clock is wall time. Simulation time is an explicit field;
            # this bridge intentionally does not own the global /clock topic.
            packet["ros_stamp"] = {"sec": stamp.sec, "nanosec": stamp.nanosec}
            self.state_pub.publish(String(data=json.dumps(packet, allow_nan=False)))
            bits = UInt8MultiArray(data=packet["active"])
            bits.layout.dim = [MultiArrayDimension(label=f"env_{packet['env_id']}:sensors", size=17, stride=17)]
            self.bits_pub.publish(bits)
            self.marker_pub.publish(make_markers(packet, stamp))

    def destroy_node(self):
        self.socket.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TactileBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
