#!/usr/bin/env -S python3 

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import pybullet as pb
import time
import pybullet_data
import numpy as np
from brian_sim import brianSim, simpleLog # <--- Import simpleLog now

from sensor_msgs.msg import JointState, Imu, LaserScan
from brian_msgs.msg import ContactDetection
from nav_msgs.msg import Odometry 
from std_srvs.srv import Empty 
from std_srvs.srv import SetBool

class brianPybullet(Node):
    def __init__(self, urdf_dir, robot_name):
        super().__init__('brian_pybullet_node')
        self.get_logger().info("brian_pybullet_node is running!")

        # PyBullet setup
        pb.setTimeStep(0.001)
        pb.setGravity(0, 0, -9.8)
        self.plane = pb.loadURDF("plane.urdf", [0, 0, 0]) 

        # Load room (obstacles)
        room_urdf_dir = "/home/nathan/brian/install/brian_pybullet/share/brian_pybullet/worlds/room.urdf"
        self.room_id = pb.loadURDF(room_urdf_dir, [0.0, 0.0, 0]) 

        pb.changeDynamics(self.plane, -1, lateralFriction=1, spinningFriction=0.01, rollingFriction=0.01)

        self.start_pos = [0, 0, 0.26] 
        self.start_ori_euler = [0, 0, 0] 
        
        # Pass the node's logger to brianSim for consistent logging
        # Or, you can use simpleLog if you prefer the print output.
        # For a ROS2 node, it's better to use the ROS2 logger.
        # Create a simpleLog instance here to pass to brianSim
        brian_sim_logger = simpleLog() # Using your simpleLog
        # brian_sim_logger = self.get_logger() # Option: Use ROS2 logger

        self.brian = brianSim(urdf_dir, self.start_pos, self.start_ori_euler, self.plane, logging=brian_sim_logger)

        # Initialize ROS 2 messages
        self.jointStateMsg = JointState()
        self.jointStateMsg.name = list(self.brian.getRevoluteJointNames()) 
        self.feet_contact = np.zeros(4, dtype='bool')
        self.contact_det_msg = ContactDetection()
        self.imu_msg = Imu() 
        self.odom_msg = Odometry() 
        self.laser_scan_msg = LaserScan() 

        # ROS 2 Publishers
        self.jointStatePub = self.create_publisher(JointState, f'/{robot_name}/joints_state', 10)
        self.robotImuPub = self.create_publisher(Imu, f'/{robot_name}/imu_data', 10)
        self.robotOdomPub = self.create_publisher(Odometry, f'/{robot_name}/odom_data', 10) 
        self.contactDetectionPub = self.create_publisher(ContactDetection, f'/{robot_name}/contact_detection', 10)
        self.laserScanPub = self.create_publisher(LaserScan, '/scan', 10)

        # ROS 2 Subscriber
        self.create_subscription(JointState, f'/{robot_name}/joints_control', self.jointControlCB, 10)
        
        # ROS 2 Service for Reset
        self.reset_service = self.create_service(Empty, '/brian/reset_sim', self.reset_sim_callback)
        self.log_state_client = None # Will be set during logging
        self.log_file_path = ""
        self.logging_service = self.create_service(SetBool, '/brian/set_logging', self.set_logging_callback)


        # Main simulation timer
        self.timer = self.create_timer(0.001, self.mainThread)


    def set_logging_callback(self, request, response):
        """ROS 2 Service callback to start/stop PyBullet state logging."""
        if request.data: # True to start logging
            if self.log_state_client is not None:
                self.get_logger().warn("PyBullet state logging already active. Stopping previous log.")
                pb.stopStateLogging(self.log_state_client)
            
            # Create a unique filename for the log
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.log_file_path = f"/tmp/brian_sim_log_{timestamp}.bullet" # Save to /tmp or a dedicated log folder
            
            self.get_logger().info(f"Starting PyBullet state logging to {self.log_file_path}")
            # pb.ER_BULLET_VISUALIZER_GUI makes it record GUI-specific events, potentially making playback smoother
            # pb.ER_BULLET_FILE_SAVE_MP4 for direct mp4, but often requires ffmpeg and specific GUI connections.
            # Using FILE_LOG for .bullet files which are playbackable.
            self.log_state_client = pb.startStateLogging(pb.STATE_LOGGING_VIDEO_MP4, self.log_file_path) # Changed to MP4
            # If MP4 causes issues, revert to: pb.startStateLogging(pb.STATE_LOGGING_GENERIC_FILE, self.log_file_path)
            # The VIDEO_MP4 option will attempt to create an MP4 directly, but note it might not always work in DIRECT mode.
            # For guaranteed MP4, you usually play back the .bullet in a GUI and record that.
            
            response.success = True
            response.message = f"Started logging to {self.log_file_path}"
        else: # False to stop logging
            if self.log_state_client is not None:
                self.get_logger().info(f"Stopping PyBullet state logging from {self.log_file_path}")
                pb.stopStateLogging(self.log_state_client)
                self.log_state_client = None
                response.success = True
                response.message = f"Stopped logging from {self.log_file_path}"
            else:
                self.get_logger().warn("PyBullet state logging not active. Cannot stop.")
                response.success = False
                response.message = "Logging not active."
        return response
    
    def reset_sim_callback(self, request, response):
        """ROS 2 Service callback to reset the simulation."""
        self.get_logger().info("Resetting PyBullet simulation...")
        
        # Call the reset_robot method of the brianSim instance
        self.brian.reset_robot()
        
        # Optionally, reset the room's position if it's dynamic
        # pb.resetBasePositionAndOrientation(self.room_id, [0.0, 0.0, 0], [0, 0, 0, 1])

        # Reset feet contact and other internal states if necessary
        self.feet_contact = np.zeros(4, dtype='bool')

        self.get_logger().info("PyBullet simulation reset complete.")
        return response 

    def jointControlCB(self, data):
        # The `velocity` argument in `setJointPosition` is not used in brian_sim.py
        # Your brian_sim.py setJointPosition only takes joint_pos.
        # You'll need to update brian_sim.py if you want to control maxVelocity via this topic.
        # For now, it uses maxVelocity=20 as hardcoded in brian_sim.setJointPosition.
        self.brian.setJointPosition(data.position)

    def mainThread(self):
        pb.stepSimulation()

        self.feet_contact = self.brian.getFeetContact()
        self.is_colliding_with_obstacle = self._check_obstacle_collision() # NEW: Check collision with room

        self.publishJointState()
        self.publishIMUdata()
        self.publishOdometryData() # NEW: Publish Odometry
        self.publishLidarData()
        self.publishContactDetectorData()

    def _check_obstacle_collision(self):
        """Checks for collisions between the robot and the loaded room obstacles."""
        contacts_with_room = pb.getContactPoints(bodyA=self.brian.robot, bodyB=self.room_id)
        return len(contacts_with_room) > 0 # True if any contact points exist

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
        self.contact_det_msg.collided_with_obstacle = self.is_colliding_with_obstacle # NEW: Add collision status
        self.contactDetectionPub.publish(self.contact_det_msg)

    def publishIMUdata(self):
        # brian.getRobotState() provides all necessary IMU components
        _, _, ang_pos, ang_vel, _, ang_acc = self.brian.getRobotState() # We need linear_acceleration for IMU

        self.imu_msg.header.stamp = self.get_clock().now().to_msg()
        self.imu_msg.header.frame_id = "base_link" # Assuming "base_link" is the IMU frame

        # Quaternion orientation
        self.imu_msg.orientation.x = ang_pos[0]
        self.imu_msg.orientation.y = ang_pos[1]
        self.imu_msg.orientation.z = ang_pos[2]
        self.imu_msg.orientation.w = ang_pos[3]
        
        # Angular velocity
        self.imu_msg.angular_velocity.x = ang_vel[0]
        self.imu_msg.angular_velocity.y = ang_vel[1]
        self.imu_msg.angular_velocity.z = ang_vel[2]
        
        # Linear acceleration (from brian_sim)
        _, _, _, _, lin_acc, _ = self.brian.getRobotState() # Call again to get current acceleration
        self.imu_msg.linear_acceleration.x = lin_acc[0]
        self.imu_msg.linear_acceleration.y = lin_acc[1]
        self.imu_msg.linear_acceleration.z = lin_acc[2]

        self.robotImuPub.publish(self.imu_msg)

    def publishOdometryData(self): # NEW FUNCTION
        """Publishes the robot's base position and linear velocity."""
        lin_pos, lin_vel, ang_pos, _, _, _ = self.brian.getRobotState()

        self.odom_msg.header.stamp = self.get_clock().now().to_msg()
        self.odom_msg.header.frame_id = "odom" # Standard for odometry frame
        self.odom_msg.child_frame_id = "base_link" # Frame of the robot's base

        self.odom_msg.pose.pose.position.x = lin_pos[0]
        self.odom_msg.pose.pose.position.y = lin_pos[1]
        self.odom_msg.pose.pose.position.z = lin_pos[2]

        self.odom_msg.pose.pose.orientation.x = ang_pos[0] # Use current orientation from brian_sim
        self.odom_msg.pose.pose.orientation.y = ang_pos[1]
        self.odom_msg.pose.pose.orientation.z = ang_pos[2]
        self.odom_msg.pose.pose.orientation.w = ang_pos[3]

        self.odom_msg.twist.twist.linear.x = lin_vel[0]
        self.odom_msg.twist.twist.linear.y = lin_vel[1]
        self.odom_msg.twist.twist.linear.z = lin_vel[2]

        # Angular velocity is already in IMU, but also common in Odometry
        # Using angular_velocity from getRobotState directly
        _, _, _, ang_vel, _, _ = self.brian.getRobotState()
        self.odom_msg.twist.twist.angular.x = ang_vel[0]
        self.odom_msg.twist.twist.angular.y = ang_vel[1]
        self.odom_msg.twist.twist.angular.z = ang_vel[2]

        self.robotOdomPub.publish(self.odom_msg)

    def publishLidarData(self):
        max_dist = 8.0 
        res_deg = 1.0 

        all_angles, distances = self.brian.getLidarData(max_dist, res_deg)

        if all_angles is None or distances is None:
            self.get_logger().error("Lidar data is None, skipping publish.")
            return

        # Ensure all_angles and distances are lists of floats
        all_angles_list = [float(a) for a in all_angles]
        # Ensure distances are valid and non-negative; replace invalid with max_dist
        distances_list = [float(d) if isinstance(d, (int, float)) and d >= 0 else max_dist for d in distances]


        self.laser_scan_msg.header.stamp = self.get_clock().now().to_msg()
        self.laser_scan_msg.header.frame_id = "lidar_link" # Ensure this matches your URDF

        self.laser_scan_msg.angle_min = float(np.min(all_angles_list))
        self.laser_scan_msg.angle_max = float(np.max(all_angles_list))
        self.laser_scan_msg.angle_increment = res_deg * (np.pi / 180.0)

        scan_frequency = 10.0 # Assuming 10 Hz scan rate
        self.laser_scan_msg.scan_time = 1.0 / scan_frequency
        self.laser_scan_msg.time_increment = self.laser_scan_msg.scan_time / len(distances_list) if distances_list else 0.0

        self.laser_scan_msg.range_min = 0.0 # Minimum detection range
        self.laser_scan_msg.range_max = max_dist # Maximum detection range

        self.laser_scan_msg.ranges = distances_list
        self.laser_scan_msg.intensities = distances_list # Often ranges are used for intensities in simple lidar

        self.laserScanPub.publish(self.laser_scan_msg)


def main(args=None):
    rclpy.init(args=args)

    robot_name = "brian"
    urdf_dir = "/home/nathan/brian/install/brian_description/share/brian_description/urdf/brian.urdf"

    node = rclpy.create_node('parameter_loader')
    node.declare_parameter('robot_name', robot_name)
    node.declare_parameter('urdf_path', urdf_dir)

    robot_name = node.get_parameter('robot_name').value
    urdf_dir = node.get_parameter('urdf_path').value

    node.destroy_node()

    # Change this line:
    pb.connect(pb.DIRECT) # <--- Changed from pb.GUI to pb.DIRECT
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