"""Separate ROS 2 subscriber: labeled tactile map plus live 3D sensor positions."""

import argparse
import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .protocol import decode


class TactileViewer(Node):
    def __init__(self):
        super().__init__("inspire_tactile_viewer")
        self.env_id = self.declare_parameter("env_id", 0).value
        topic = self.declare_parameter("topic", "/inspire/tactile").value
        self.packet = None
        self.received_at = 0.0
        self.create_subscription(String, topic, self.receive, 10)

    def receive(self, message):
        try:
            packet = decode(message.data)
        except (ValueError, TypeError, KeyError):
            self.get_logger().warning("Ignoring invalid tactile ROS message", throttle_duration_sec=5)
            return
        if packet["env_id"] == self.env_id:
            self.packet = packet
            self.received_at = time.monotonic()


class TactilePlot:
    def __init__(self):
        import matplotlib.pyplot as plt

        self.figure = plt.figure(figsize=(13, 7))
        self.map_ax = self.figure.add_subplot(121)
        self.xyz_ax = self.figure.add_subplot(122, projection="3d")
        self.figure.subplots_adjust(left=0.04, right=0.97, top=0.86, bottom=0.09, wspace=0.1)
        # A diagram, not measured geometry; IDs stay legible when fingers curl.
        self.xy = [(2.2, 0.4), (-0.25, 0.6), (-0.6, 1.3), (-0.95, 2.0), (-1.3, 2.7)]
        for x, top in ((0.65, 4.25), (1.85, 4.55), (3.05, 4.25), (4.25, 3.65)):
            self.xy.extend((x, top - 1.5 + i * 0.75) for i in range(3))
        ax = self.map_ax
        ax.plot([0.2, 0.2, 4.7, 4.7, 0.2], [1.5, -0.15, -0.15, 1.5, 1.5], color="#aab5bd")
        for indices in ((1, 2, 3, 4), (5, 6, 7), (8, 9, 10), (11, 12, 13), (14, 15, 16)):
            ax.plot([self.xy[i][0] for i in indices], [self.xy[i][1] for i in indices], color="#aab5bd", lw=10, alpha=0.25)
        self.dots = ax.scatter(*zip(*self.xy), s=400, c=["#aab5bd"] * 17, zorder=3)
        self.force_labels = []
        for i, (x, y) in enumerate(self.xy):
            ax.text(x, y, str(i), ha="center", va="center", fontsize=10, zorder=4)
            self.force_labels.append(ax.text(x, y - 0.28, "-- N", ha="center", fontsize=8))
        for x, y, text in [(-1.2, 3.3, "Thumb"), (0.65, 4.7, "Index"), (1.85, 5, "Middle"),
                            (3.05, 4.7, "Ring"), (4.25, 4.15, "Little"), (2.2, -0.45, "Palm")]:
            ax.text(x, y, text, ha="center", fontsize=10)
        ax.set(xlim=(-1.85, 5), ylim=(-0.8, 5.3), title="Sensor ID map (schematic)")
        ax.set_aspect("equal")
        ax.axis("off")
        self.status = self.figure.suptitle("Waiting for ROS 2 tactile data...", fontsize=13)
        self.footer = self.figure.text(0.5, 0.025, "Green = ON | gray = OFF | orange = stale / unknown", ha="center")

    def update(self, packet, stale=False):
        if packet is None:
            return
        colors = ["#e5aa45" if stale else "#29cc71" if bit else "#aab5bd" for bit in packet["active"]]
        self.dots.set_color(colors)
        for label, force in zip(self.force_labels, packet["norms_N"]):
            label.set_text("-- N" if stale else f"{force:.3f} N")
        active = [str(i) for i, bit in enumerate(packet["active"]) if bit]
        self.status.set_text(
            f"Env {packet['env_id']} | {packet['phase']} | sim {packet['time_s']:.2f} s | "
            + ("virtual regions | " if packet.get("force_source") == "contact_region_projection" else "raw bodies | ")
            + ("STALE - contact state unknown" if stale else f"ON: {', '.join(active) or 'none'}")
        )
        ax = self.xyz_ax
        elevation, azimuth = ax.elev, ax.azim
        ax.clear()
        ax.view_init(elev=elevation, azim=azimuth)
        points = packet["positions_w_m"]
        ax.scatter(*zip(*points), c=colors, s=55, depthshade=False)
        for i, xyz in enumerate(points):
            ax.text(*xyz, str(i), fontsize=8)
        center = [sum(p[k] for p in points) / 17 for k in range(3)]
        radius = max(0.06, max(max(p[k] for p in points) - min(p[k] for p in points) for k in range(3)) * 0.6)
        ax.set(xlim=(center[0] - radius, center[0] + radius),
               ylim=(center[1] - radius, center[1] + radius), zlim=(center[2] - radius, center[2] + radius),
               xlabel="world X [m]", ylabel="world Y [m]", zlabel="world Z [m]",
               title="Sensor link origins (NOT collision surface points)")
        ax.set_box_aspect((1, 1, 1))
        self.footer.set_text(f"Green = ON (norm >= {packet['threshold_N']:g} N) | gray = OFF | orange = stale / unknown")
        self.figure.canvas.draw_idle()


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=0, help="Wall seconds to run; 0 = until closed")
    parser.add_argument("--output", type=Path, help="Save the final view as PNG (also works headless)")
    options, ros_args = parser.parse_known_args(args)
    if not math.isfinite(options.duration) or options.duration < 0 or (options.headless and options.duration <= 0):
        parser.error("headless needs --duration > 0; duration must be non-negative")
    import matplotlib
    if options.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rclpy.init(args=ros_args)
    node = TactileViewer()
    plot = TactilePlot()
    if not options.headless:
        plt.show(block=False)
    start = time.monotonic()
    drawn_at = 0.0
    try:
        while rclpy.ok() and (options.duration == 0 or time.monotonic() - start < options.duration):
            if not options.headless and not plt.fignum_exists(plot.figure.number):
                break
            rclpy.spin_once(node, timeout_sec=0.02)
            now = time.monotonic()
            if now - drawn_at >= 0.1:
                plot.update(node.packet, stale=now - node.received_at > 1.0)
                drawn_at = now
            if not options.headless:
                plt.pause(0.001)
        if options.output:
            options.output.parent.mkdir(parents=True, exist_ok=True)
            plot.figure.savefig(options.output, dpi=140)
        if options.headless and node.packet is None:
            raise RuntimeError("No ROS tactile message received; check bridge, port, topic and ROS_DOMAIN_ID")
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(plot.figure)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
