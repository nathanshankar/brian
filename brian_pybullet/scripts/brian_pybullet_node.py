#!/usr/bin/env -S python3
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import pybullet as pb
import time
import pybullet_data
import numpy as np
from brian_sim import brianSim, simpleLog # <--- Import simpleLog now

# ROS 2 Message Imports
from sensor_msgs.msg import JointState, Imu, LaserScan, Image # Added Image
from brian_msgs.msg import ContactDetection
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from std_srvs.srv import SetBool
from geometry_msgs.msg import Quaternion, Vector3 # For Imu and Odometry

# OpenCV and CvBridge for image processing
import cv2 # Ensure you have opencv-python installed (pip install opencv-python)
from cv_bridge import CvBridge, CvBridgeError # Added CvBridge and CvBridgeError

class brianPybullet(Node):
    def __init__(self, urdf_dir, robot_name):
        super().__init__('brian_pybullet_node')
        self.get_logger().info("brian_pybullet_node is running!")

        # PyBullet setup
        # Use p.GUI if you want a visual window, otherwise p.DIRECT for headless
        # For video recording via ROS, p.DIRECT is usually sufficient and preferred for performance.
        self.physics_client = pb.connect(pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setTimeStep(0.001)
        pb.setGravity(0, 0, -9.8)

        self.plane = pb.loadURDF("plane.urdf", [0, 0, 0])
        self.plane_id = self.plane # Store for potential contact checks later

        # Load room (obstacles)
        room_urdf_dir = "/root/Documents/brian_ws/install/brian_pybullet/share/brian_pybullet/worlds/room.urdf"
        self.room_id = pb.loadURDF(room_urdf_dir, [0.0, 0.0, 0])

        pb.changeDynamics(self.plane, -1, lateralFriction=1, spinningFriction=0.01, rollingFriction=0.01)

        self.start_pos = [0, 0, 0.26]
        self.start_ori_euler = [0, 0, 0]

        # Pass the node's logger to brianSim for consistent logging
        brian_sim_logger = simpleLog() # Using your simpleLog, assuming it's available
        # Alternatively, if simpleLog is just a print wrapper, consider using:
        # brian_sim_logger = self.get_logger() # Option: Use ROS2 logger for brianSim messages

        self.brian = brianSim(urdf_dir, self.start_pos, self.start_ori_euler, self.plane, logging=brian_sim_logger)

        # Initialize ROS 2 messages
        self.jointStateMsg = JointState()
        self.jointStateMsg.name = list(self.brian.getRevoluteJointNames())
        self.feet_contact = np.zeros(4, dtype='bool')
        self.contact_det_msg = ContactDetection()
        self.imu_msg = Imu()
        self.odom_msg = Odometry()
        self.laser_scan_msg = LaserScan()

        # --- Camera Setup ---
        self.bridge = CvBridge() # Initialize CvBridge
        self.camera_width = 320 # Standard width
        self.camera_height = 240 # Standard height
        self.camera_fov = 90 # Field of View
        self.camera_near = 0.01
        self.camera_far = 100

        # Define camera position and target relative to the robot's base
        # You'll need to adjust these for the best view of your robot
        self.camera_offset_xyz = [0.2, 0.0, 0.3] # Offset from base_link (x,y,z)
        self.camera_target_offset_xyz = [0.5, 0.0, 0.0] # Relative target point from camera origin

        # ROS 2 Publishers
        self.jointStatePub = self.create_publisher(JointState, f'/{robot_name}/joints_state', 10)
        self.robotImuPub = self.create_publisher(Imu, f'/{robot_name}/imu_data', 10)
        self.robotOdomPub = self.create_publisher(Odometry, f'/{robot_name}/odom_data', 10)
        self.contactDetectionPub = self.create_publisher(ContactDetection, f'/{robot_name}/contact_detection', 10)
        self.laserScanPub = self.create_publisher(LaserScan, '/scan', 10)
        self.cameraImagePub = self.create_publisher(Image, f'/{robot_name}/camera/image_raw', 10) # New Image Publisher

        # ROS 2 Subscriber
        self.create_subscription(JointState, f'/{robot_name}/joints_control', self.jointControlCB, 10)

        # ROS 2 Service for Reset
        self.reset_service = self.create_service(Empty, '/brian/reset_sim', self.reset_sim_callback)
        self.current_log_id = -1 # PyBullet logging ID
        self.log_file_path = ""
        self.logging_service = self.create_service(SetBool, '/brian/set_logging', self.set_logging_callback)

        # Main simulation timer
        self.timer = self.create_timer(0.001, self.mainThread) # Runs at 1000 Hz, PyBullet timestep is 0.001s

    def set_logging_callback(self, request, response):
        """ROS 2 Service callback to start/stop PyBullet state logging."""
        if request.data: # True to start logging
            if self.current_log_id != -1:
                self.get_logger().warn("PyBullet state logging already active. Stopping previous log.")
                pb.stopStateLogging(self.current_log_id)

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.log_file_path = f"/tmp/brian_sim_log_{timestamp}.bullet" # Save to /tmp or a dedicated log folder

            self.get_logger().info(f"Starting PyBullet state logging to {self.log_file_path}")
            # Use STATE_LOGGING_GENERIC_FILE for .bullet logs.
            # RecordVideo in Gymnasium will handle the MP4 conversion from the ROS image stream.
            self.current_log_id = pb.startStateLogging(pb.STATE_LOGGING_GENERIC_FILE, self.log_file_path)

            response.success = True
            response.message = f"Started logging to {self.log_file_path}"
        else: # False to stop logging
            if self.current_log_id != -1:
                self.get_logger().info(f"Stopping PyBullet state logging from {self.log_file_path}")
                pb.stopStateLogging(self.current_log_id)
                self.current_log_id = -1
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

        # Reset feet contact and other internal states if necessary
        self.feet_contact = np.zeros(4, dtype='bool')

        self.get_logger().info("PyBullet simulation reset complete.")
        return response

    def jointControlCB(self, data):
        # Your brian_sim.py setJointPosition only takes joint_pos.
        # It uses maxVelocity=20 as hardcoded in brian_sim.setJointPosition.
        self.brian.setJointPosition(data.position)

    def mainThread(self):
        pb.stepSimulation()
        current_time = self.get_clock().now().to_msg() # Get current ROS time once per loop

        self.feet_contact = self.brian.getFeetContact()
        self.is_colliding_with_obstacle = self._check_obstacle_collision()

        self.publishJointState(current_time)
        self.publishIMUdata(current_time)
        self.publishOdometryData(current_time)
        self.publishLidarData(current_time)
        self.publishContactDetectorData(current_time)
        self.publishCameraImage(current_time) # NEW: Publish camera image

    def _check_obstacle_collision(self):
        """Checks for collisions between the robot and the loaded room obstacles."""
        contacts_with_room = pb.getContactPoints(bodyA=self.brian.robot, bodyB=self.room_id)
        return len(contacts_with_room) > 0 # True if any contact points exist

    def publishJointState(self, current_time):
        pos, vel, torq = self.brian.getMotorJointStates()
        self.jointStateMsg.position = pos
        self.jointStateMsg.velocity = vel
        self.jointStateMsg.effort = torq
        self.jointStateMsg.header.stamp = current_time
        self.jointStatePub.publish(self.jointStateMsg)

    def publishContactDetectorData(self, current_time):
        self.contact_det_msg.header.stamp = current_time
        self.contact_det_msg.feet_stance = [bool(self.feet_contact[0]), bool(self.feet_contact[1]), bool(self.feet_contact[2]), bool(self.feet_contact[3])]
        self.contact_det_msg.collided_with_obstacle = self.is_colliding_with_obstacle
        self.contactDetectionPub.publish(self.contact_det_msg)

    def publishIMUdata(self, current_time):
        _, _, ang_pos, ang_vel, lin_acc, _ = self.brian.getRobotState()

        self.imu_msg.header.stamp = current_time
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

        # Linear acceleration
        self.imu_msg.linear_acceleration.x = lin_acc[0]
        self.imu_msg.linear_acceleration.y = lin_acc[1]
        self.imu_msg.linear_acceleration.z = lin_acc[2]

        self.robotImuPub.publish(self.imu_msg)

    def publishOdometryData(self, current_time):
        lin_pos, lin_vel, ang_pos_quat, ang_vel_body, _, _ = self.brian.getRobotState()

        self.odom_msg.header.stamp = current_time
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = "base_link"

        self.odom_msg.pose.pose.position.x = lin_pos[0]
        self.odom_msg.pose.pose.position.y = lin_pos[1]
        self.odom_msg.pose.pose.position.z = lin_pos[2]

        self.odom_msg.pose.pose.orientation.x = ang_pos_quat[0]
        self.odom_msg.pose.pose.orientation.y = ang_pos_quat[1]
        self.odom_msg.pose.pose.orientation.z = ang_pos_quat[2]
        self.odom_msg.pose.pose.orientation.w = ang_pos_quat[3]

        self.odom_msg.twist.twist.linear.x = lin_vel[0]
        self.odom_msg.twist.twist.linear.y = lin_vel[1]
        self.odom_msg.twist.twist.linear.z = lin_vel[2]

        self.odom_msg.twist.twist.angular.x = ang_vel_body[0]
        self.odom_msg.twist.twist.angular.y = ang_vel_body[1]
        self.odom_msg.twist.twist.angular.z = ang_vel_body[2]

        self.robotOdomPub.publish(self.odom_msg)

    def publishLidarData(self, current_time):
        max_dist = 8.0
        res_deg = 1.0

        all_angles, distances = self.brian.getLidarData(max_dist, res_deg)

        if all_angles is None or distances is None:
            self.get_logger().error("Lidar data is None, skipping publish.")
            return

        all_angles_list = [float(a) for a in all_angles]
        distances_list = [float(d) if isinstance(d, (int, float)) and d >= 0 else max_dist for d in distances]


        self.laser_scan_msg.header.stamp = current_time
        self.laser_scan_msg.header.frame_id = "lidar_link" # Ensure this matches your URDF

        self.laser_scan_msg.angle_min = float(np.min(all_angles_list))
        self.laser_scan_msg.angle_max = float(np.max(all_angles_list))
        self.laser_scan_msg.angle_increment = res_deg * (np.pi / 180.0)

        scan_frequency = 10.0 # Assuming 10 Hz scan rate
        self.laser_scan_msg.scan_time = 1.0 / scan_frequency
        self.laser_scan_msg.time_increment = self.laser_scan_msg.scan_time / len(distances_list) if distances_list else 0.0

        self.laser_scan_msg.range_min = 0.0
        self.laser_scan_msg.range_max = max_dist

        self.laser_scan_msg.ranges = distances_list
        self.laser_scan_msg.intensities = distances_list # Often ranges are used for intensities in simple lidar

        self.laserScanPub.publish(self.laser_scan_msg)

    def publishCameraImage(self, current_time):
        # Get robot's base position (we'll look at this point)
        base_pos, _ = pb.getBasePositionAndOrientation(self.brian.robot)

        # --- Define fixed camera position in world coordinates ---
        # Example: Camera 2 meters behind, 0 meters left/right, 1 meter above the ground
        # This will be a static camera position relative to the world origin.
        camera_eye_position = [base_pos[0] - 2.0, base_pos[1], base_pos[2] + 1.0] # Behind and above the robot

        # --- Define what the camera is looking at ---
        # We want it to look at the robot's base
        camera_target_position = list(base_pos) # Convert tuple to list for modification if needed
        # Optionally, adjust target height to look slightly above the robot's feet
        # camera_target_position[2] += 0.1 # Look slightly at the robot's body, not just ground

        # Compute view matrix
        view_matrix = pb.computeViewMatrix(
            cameraEyePosition=camera_eye_position,
            cameraTargetPosition=camera_target_position,
            cameraUpVector=[0, 0, 1] # Z-axis is up
        )
        projection_matrix = pb.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=float(self.camera_width) / self.camera_height,
            nearVal=self.camera_near,
            farVal=self.camera_far
        )

        img_arr = pb.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=pb.ER_TINY_RENDERER # or pb.ER_BULLET_HARDWARE_OPENGL for faster rendering if available
        )

        rgb_image = np.reshape(img_arr[2], (self.camera_height, self.camera_width, 4))[:, :, :3]

        try:
            ros_image_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
            ros_image_msg.header.stamp = current_time
            ros_image_msg.header.frame_id = 'camera_link' # A frame for your camera sensor
            self.cameraImagePub.publish(ros_image_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")

    def destroy_node(self):
        self.get_logger().info("PyBullet disconnected.")
        pb.disconnect() # Disconnect from PyBullet server
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    robot_name = "brian"
    # Ensure this path is correct for your setup
    urdf_dir = get_package_share_directory('brian_description') + "/urdf/brian.urdf"

    # Minimal setup for parameter loading (if you actually use parameters from launch files)
    node = rclpy.create_node('parameter_loader_temp') # Temporary node for parameters
    node.declare_parameter('robot_name', robot_name)
    node.declare_parameter('urdf_path', urdf_dir)
    robot_name = node.get_parameter('robot_name').value
    urdf_dir = node.get_parameter('urdf_path').value
    node.destroy_node() # Destroy temporary node

    brian_node = brianPybullet(urdf_dir, robot_name)

    try:
        rclpy.spin(brian_node)
    except KeyboardInterrupt:
        pass
    finally:
        brian_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()