#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

def main():
    rclpy.init()
    node = rclpy.create_node('trajectory_client')
    client = ActionClient(node, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')
    
    if not client.wait_for_server(timeout_sec=5.0):
        node.get_logger().error("Action server not available!")
        return

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = [
        'left_thumb_1_joint', 'left_thumb_2_joint', 
        'left_index_1_joint', 'left_middle_1_joint',
        'left_ring_1_joint', 'left_little_1_joint'
    ]
    
    point1 = JointTrajectoryPoint()
    point1.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    point1.velocities = [3.5, 3.5, 3.5, 3.5, 3.5, 3.5]
    point1.time_from_start.sec = 1
    
    point2 = JointTrajectoryPoint()
    point2.positions = [1.2, 0.3, 0.9, 0.6, 0.4, 0.2]
    point2.velocities = [3.5, 3.5, 3.5, 3.5, 3.5, 3.5]
    point2.time_from_start.sec = 5
    
    goal.trajectory.points = [point1, point2]
    
    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
