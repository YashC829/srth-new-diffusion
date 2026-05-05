#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time
import argparse
import sys
import crtk
import dvrk
import math
import numpy as np
import PyKDL

# Local imports
from rostopics_ros2 import RosTopics

class example_application:
    def __init__(self, ral, arm_name, expected_interval):
        # In ROS 2, printing is replaced by logging through the node handle
        self.ral = ral
        self.node = ral._node # Access the underlying rclpy node
        self.node.get_logger().info(f'Configuring dvrk_arm_test for {arm_name}')
        
        self.expected_interval = expected_interval
        
        # Initialize the dVRK PSM handle using the RAL object
        self.arm = dvrk.psm(ral=ral,
                            arm_name=arm_name)
        self.arm_name = arm_name

    def home(self):
        self.arm.check_connections()
        self.node.get_logger().info('Starting enable')
        if not self.arm.enable(10):
            sys.exit('Failed to enable within 10 seconds')
        
        self.node.get_logger().info('Starting home')
        if not self.arm.home(10):
            sys.exit('Failed to home within 10 seconds')
            
        goal = np.copy(self.arm.setpoint_jp())
        goal.fill(0)
        if self.arm.name() in ['PSM1', 'PSM2', 'PSM3', 'ECM']:
            goal[2] = 0.12
            
        self.node.get_logger().info('Moving to starting position')
        self.arm.move_jp(goal).wait()
        self.node.get_logger().info('Home complete')

    def run_servo_jp(self):
        self.node.get_logger().info('Starting servo_jp')
        initial_joint_position = np.copy(self.arm.setpoint_jp())
        amplitude = math.radians(5.0)
        duration = 5
        samples = duration / self.expected_interval
        
        goal_p = np.copy(initial_joint_position)
        goal_v = np.zeros(goal_p.size)
        start = time.time()

        # In ROS 2, we create the rate from the node
        sleep_rate = self.node.create_rate(1.0 / self.expected_interval)
        for i in range(int(samples)):
            angle = i * math.radians(360.0) / samples
            goal_p[0] = initial_joint_position[0] + amplitude * (1.0 - math.cos(angle))
            goal_v[0] = amplitude * math.sin(angle)
            self.arm.servo_jp(goal_p, goal_v)
            sleep_rate.sleep()

    def move_robot_in_direction(self, direction, current_pose, current_jaw_position):
        initial_cartesian_position = PyKDL.Frame()
        print("Current pose: ", current_pose)
        initial_cartesian_position.p = PyKDL.Vector(current_pose.position.x, current_pose.position.y, current_pose.position.z)
        initial_cartesian_position.M = PyKDL.Rotation.Quaternion(current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w)
        initial_jaw_position = current_jaw_position
        goal = PyKDL.Frame()
        goal.p = PyKDL.Vector(current_pose.position.x, current_pose.position.y, current_pose.position.z)
        goal.M = PyKDL.Rotation.Quaternion(current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w)
        
        translation = ['up', 'down', 'left', 'right', 'forward', 'backward']
        jaw_motion = ['open', 'close']
        if direction in translation:
            if direction == 'up':
                goal.p = initial_cartesian_position.p
                goal.p[1] += 0.001
                
            if direction == 'down':
                goal.p = initial_cartesian_position.p
                goal.p[1] -= 0.001
                
            if direction == 'left':
                goal.p = initial_cartesian_position.p
                goal.p[0] += 0.001
                
            if direction == 'right':
                goal.p = initial_cartesian_position.p
                goal.p[0] -= 0.001
                
            if direction == 'forward':
                goal.p = initial_cartesian_position.p
                goal.p[2] += 0.001
                
            if direction == 'backward':
                goal.p = initial_cartesian_position.p
                goal.p[2] -= 0.001
                
            self.arm.servo_cp(goal)
            self.arm.jaw.servo_jp(initial_jaw_position)
            
        elif direction in jaw_motion:
            if direction == 'open':
                self.arm.servo_cp(initial_cartesian_position)
                
                self.arm.jaw.open(angle = 0.6).wait()
                
            if direction == 'close':
                self.arm.servo_cp(initial_cartesian_position)
                
                self.arm.jaw.open(angle = -0.5).wait()
        
    def move_robot_to_pose(self, pose):
        goal = PyKDL.Frame()
        goal.p = PyKDL.Vector(pose[0], pose[1], pose[2])
        goal.M = PyKDL.Rotation.Quaternion(pose[3], pose[4], pose[5], pose[6])

        self.arm.move_cp(goal).wait()
    # def run_full_pose_goal(self, wp):
    #     pos_diff_thres = 0.003
    #     jaw_diff_thres = 0.5
    #     self.arm.check_connections()
    #     start, ts = self.arm.setpoint_cp()
    #     print('Getting initial position', start)
    #     while ts == 0:
    #         start, ts = self.arm.setpoint_cp()
    #         time.sleep(0.01)
    #         print('waiting for data')

    #     try:

    #         initial_cp = self.arm.setpoint_cp()[0]
    #         jaw_position = self.arm.jaw.setpoint_jp()[0]
    #     except Exception as e:
    #         self.node.get_logger().error(f"Could not get robot state: {e}")
    #         return # Avoid crashing the thread

        
    #     print(initial_cp)
    #     print(jaw_position)
    #     goal = PyKDL.Frame()
    #     goal.p = PyKDL.Vector(wp[0], wp[1], wp[2])
    #     goal.M = PyKDL.Rotation.Quaternion(wp[3], wp[4], wp[5], wp[6])

    #     # Calculate distance
    #     init_arr = np.array([initial_cp.p[0], initial_cp.p[1], initial_cp.p[2]])
    #     diff_norm = np.linalg.norm(wp[0:3] - init_arr)
    #     jaw_diff = wp[-1] - jaw_position[0]

    #     if abs(diff_norm) > pos_diff_thres:
    #         self.arm.move_cp(goal).wait()
    #         print("moved")
    #     else:
    #         self.arm.servo_cp(goal)
    #         print("servoed")

    #     if abs(jaw_diff) > jaw_diff_thres:
    #         self.arm.jaw.open(angle=wp[-1])
    #         print("opened")

    #     else:
    #         jaw_position[0] = wp[-1]
    #         self.arm.jaw.servo_jp(jaw_position)
    #         print("servoed jaw")

    def run_full_pose_goal(self, wp, jaw_angle=None):
        jaw_diff_thres = 0.3
        goal = PyKDL.Frame()
        goal.p = PyKDL.Vector(wp[0], wp[1], wp[2])
        goal.M = PyKDL.Rotation.Quaternion(wp[3], wp[4], wp[5], wp[6])

        self.arm.servo_cp(goal)

        # self.arm.jaw.servo_jp(np.array([wp[-1]]))
        # print(jaw_angle.position[0])
        # print(f"Desired jaw: {wp[-1]}, Current jaw: {jaw_angle.position[0] if jaw_angle is not None else 'N/A'}")
        jaw_diff = abs(wp[-1] - jaw_angle.position[0]) if jaw_angle is not None else 0
        # print(f"Moving jaw towards target. Current: {jaw_angle.position[0]}, Target: {wp[-1]}")
        if abs(jaw_diff) > jaw_diff_thres:
            jaw_angle.position[0] = wp[-1] + jaw_diff_thres * np.sign(jaw_diff) # Move in the direction of the target but only by the threshold amount
        # self.arm.jaw.open(angle=wp[-1])

        # else:
            # jaw_position[0] = wp[-1]
        self.arm.jaw.servo_jp(np.array([wp[-1]]))
        # print("servoed jaw")

    def run_jaw_servo(self):
        self.node.get_logger().info('Starting jaw servo')
        start_angle = math.radians(-0.6)
        self.arm.jaw.open(angle=start_angle).wait()

def main():
    # ROS 2 Initialization logic
    argv = crtk.ral.parse_argv(sys.argv[1:])
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arm', type=str, required=True,
                        choices=['ECM', 'MTML', 'MTMR', 'PSM1', 'PSM2', 'PSM3'])
    parser.add_argument('-i', '--interval', type=float, default=0.01)
    args = parser.parse_args(argv)

    # 1. Create the RAL object (this initializes rclpy and the node)
    ral = crtk.ral('dvrk_arm_control_node')
    
    # 2. Initialize application
    application = ExampleApplication(ral, args.arm, args.interval)
    
    # 3. Use RAL to spin and execute the specific task
    try:
        ral.spin_and_execute(application.run_jaw_servo)
    except KeyboardInterrupt:
        pass
    finally:
        # RAL handles rclpy.shutdown() internally
        pass

if __name__ == '__main__':
    main()