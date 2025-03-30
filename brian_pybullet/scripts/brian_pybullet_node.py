#!/usr/bin/env -S python3 

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import pybullet as pb
import time
import pybullet_data
import numpy as np
from brian_sim import brianSim

from sensor_msgs.msg import JointState, Imu, LaserScan
from brian_msgs.msg import ContactDetection


class brianPybullet(Node):
    def __init__(self, urdf_dir, robot_name):
        super().__init__('brian_pybullet_node')
        self.get_logger().info("brian_pybullet_node is running!")

        pb.setTimeStep(0.001)
        pb.setGravity(0, 0, -9.8)
        ground = pb.loadURDF("plane.urdf", [0, 0, 0])

        room_urdf_dir = "/home/nathan/brian/install/brian_pybullet/share/brian_pybullet/worlds/room.urdf"
        self.get_logger().error(room_urdf_dir)
        room = pb.loadURDF(room_urdf_dir, [0.0, 0.0, 0])

        pb.changeDynamics(ground, -1, lateralFriction=1, spinningFriction=0.01, rollingFriction=0.01)

        startpos = [0, 0, 0.26] 
        startOri = [0, 0, 0] 
        self.brian = brianSim(urdf_dir, startpos, startOri, ground)
        self.jointStateMsg = JointState()
        self.jointStateMsg.name = list(self.brian.getRevoluteJointNames())
        self.feet_contact = np.zeros(4, dtype='bool')
        self.contact_det_msg = ContactDetection()

        self.jointStatePub = self.create_publisher(JointState, f'/{robot_name}/joints_state', 10)
        self.robotImuPub = self.create_publisher(Imu, f'/{robot_name}/imu_data', 10)
        self.contactDetectionPub = self.create_publisher(ContactDetection, f'/{robot_name}/contact_detection', 10)
        self.laserScanPub = self.create_publisher(LaserScan, '/scan', 10)

        self.create_subscription(JointState, f'/{robot_name}/joints_control', self.jointControlCB, 10)
        self.timer = self.create_timer(0.001, self.mainThread)

    def jointControlCB(self, data):
        self.brian.setJointPosition(data.position, velocity=1)

    def mainThread(self):
        pb.stepSimulation()

        self.feet_contact = self.brian.getFeetContact()

        self.publishJointState()
        self.publishIMUdata()
        self.publishLidarData()
        self.publishContactDetectorData()

    def publishJointState(self):
        pos, vel, torq = self.brian.getMotorJointStates()
        self.jointStateMsg.position = pos
        self.jointStateMsg.velocity = vel
        self.jointStateMsg.effort = torq
        self.jointStateMsg.header.stamp = self.get_clock().now().to_msg()
        self.jointStatePub.publish(self.jointStateMsg)

    def publishContactDetectorData(self):
        self.contact_det_msg.header.stamp = self.get_clock().now().to_msg()
        self.contact_det_msg.feet_stance = [bool(self.feet_contact[0]), bool(self.feet_contact[1]), bool(self.feet_contact[2]), bool(self.feet_contact[3])]
        self.contactDetectionPub.publish(self.contact_det_msg)

    def publishIMUdata(self):
        pos, vel, ang_pos, ang_vel, acc, ang_acc = self.brian.getRobotState()

        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = "base_link"

        imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w = ang_pos
        imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z = ang_vel
        imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z = acc

        self.robotImuPub.publish(imu_msg)

    def publishLidarData(self):
        max_dist = 8.0 
        res_deg = 1.0 

        all_angles, distances = self.brian.getLidarData(max_dist, res_deg)

        if all_angles is None or distances is None:
            self.get_logger().error("Lidar data is None, skipping publish.")
            return

        all_angles = [float(a) for a in all_angles if isinstance(a, (int, float))]
        distances = [float(d) if isinstance(d, (int, float)) and d > 0 else 0.0 for d in distances]

        lidar_msg = LaserScan()
        lidar_msg.angle_min = 0.0
        lidar_msg.angle_max = 2 * np.pi
        lidar_msg.angle_increment = res_deg * (np.pi / 180.0)

        scan_frequency = 10.0  
        lidar_msg.scan_time = 1.0 / scan_frequency
        lidar_msg.time_increment = lidar_msg.scan_time / len(distances) if distances else 0.0

        lidar_msg.range_min = 0.0
        lidar_msg.range_max = max_dist

        if not distances:
            self.get_logger().warn("Lidar data is empty, setting default range values.")
            lidar_msg.ranges = [0.0]
            lidar_msg.intensities = [0.0]
        else:
            lidar_msg.ranges = list(np.array(distances, dtype=np.float32))
            lidar_msg.intensities = list(np.array(distances, dtype=np.float32)) 

        lidar_msg.header.stamp = self.get_clock().now().to_msg()
        lidar_msg.header.frame_id = "lidar_link"

        self.get_logger().info("Publishing Lidar Data.")
        self.laserScanPub.publish(lidar_msg)


def main(args=None):
    rclpy.init(args=args)

    robot_name = "brian"
    urdf_dir = "/home/nathan/brian/install/brian_description/share/brian_description/urdf/brian.urdf"

    node = rclpy.create_node('parameter_loader')
    node.declare_parameter('robot_name', robot_name)
    node.declare_parameter('urdf_path', urdf_dir)

    if node.has_parameter('robot_name'):
        robot_name = node.get_parameter('robot_name').value

    if node.has_parameter('urdf_path'):
        urdf_dir = node.get_parameter('urdf_path').value

    node.destroy_node()

    pb.connect(pb.GUI) 
    pb.setAdditionalSearchPath(pybullet_data.getDataPath()) 

    brian_node = brianPybullet(urdf_dir, robot_name)

    try:
        rclpy.spin(brian_node)
    except KeyboardInterrupt:
        pass
    finally:
        brian_node.destroy_node()
        rclpy.shutdown()
        pb.disconnect()


if __name__ == '__main__':
    main()
