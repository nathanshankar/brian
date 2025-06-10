# brian_gym_env.py
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

# ROS 2 message imports
from sensor_msgs.msg import JointState, Imu, LaserScan, Image
from brian_msgs.msg import ContactDetection
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

# OpenCV and CvBridge for image processing
import cv2
from cv_bridge import CvBridge, CvBridgeError

class BrianGymEnv(gym.Env):
    """
    OpenAI Gym environment for the Brian quadruped robot in PyBullet,
    communicating via ROS 2 topics.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    # Added node_id to the constructor for unique node names
    def __init__(self, robot_name='brian', render_mode=None, node_id=0):
        # Initialize ROS 2 communication within the environment
        # Use node_id for a unique node name to avoid "Publisher already registered" warnings
        super().__init__('brian_gym_env_node_' + str(node_id))
        self.ros_node = self # The Gym environment IS the ROS node
        self.callback_group = ReentrantCallbackGroup()

        self.render_mode = render_mode
        self.robot_name = robot_name

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Initialize sensor data storage
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.joint_torques = np.zeros(12)
        self.base_position = np.zeros(3)
        self.base_orientation = np.zeros(4)
        self.base_linear_velocity = np.zeros(3)
        self.base_angular_velocity = np.zeros(3)
        self.base_linear_acceleration = np.zeros(3)
        self.feet_contact = np.zeros(4, dtype=bool)
        self.collided_with_obstacle = False
        self.lidar_ranges = np.full(360, 8.0) # Assuming 360-degree LiDAR, max range 8.0
        self.latest_camera_image = None # For RGB image

        # Expected dimensions for observation space
        self.num_joints = 12
        self.num_lidar_points = 360
        self.expected_camera_height = 480
        self.expected_camera_width = 640

        # ROS 2 Subscribers
        self.create_subscription(JointState, f'/{self.robot_name}/joints_state', self.joint_state_callback, qos_profile, callback_group=self.callback_group)
        self.create_subscription(Imu, f'/{self.robot_name}/imu_data', self.imu_callback, qos_profile, callback_group=self.callback_group)
        self.create_subscription(Odometry, f'/{self.robot_name}/odom_data', self.odometry_callback, qos_profile, callback_group=self.callback_group)
        self.create_subscription(ContactDetection, f'/{self.robot_name}/contact_detection', self.contact_detection_callback, qos_profile, callback_group=self.callback_group)
        self.create_subscription(LaserScan, '/scan', self.laser_scan_callback, qos_profile, callback_group=self.callback_group)
        self.create_subscription(Image, f'/{self.robot_name}/camera/image_raw', self.camera_image_callback, qos_profile, callback_group=self.callback_group)

        # ROS 2 Publisher for joint commands
        self.joint_command_publisher = self.create_publisher(JointState, f'/{self.robot_name}/joints_control', 10)

        # ROS 2 Service Client for simulation reset
        self.reset_sim_client = self.create_client(Empty, '/brian/reset_sim', callback_group=self.callback_group)
        while not self.reset_sim_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info('reset_sim service not available, waiting again...') # Use self.get_logger()

        # ROS 2 Executor for callbacks
        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self)

        # Initialize CvBridge for camera image processing
        self.bridge = CvBridge()

        # Define observation space
        # Joint positions (12), joint velocities (12), base position (3), base orientation (4),
        # base linear velocity (3), base angular velocity (3), base linear acceleration (3),
        # feet contact (4), LiDAR ranges (360), camera image (480*640*3)
        obs_low = np.array([-np.inf] * (12 + 12 + 3 + 4 + 3 + 3 + 3 + 4 + 360))
        obs_high = np.array([np.inf] * (12 + 12 + 3 + 4 + 3 + 3 + 3 + 4 + 360))
        self.observation_space = spaces.Dict({
            "joint_state": spaces.Box(low=-np.pi, high=np.pi, shape=(12,), dtype=np.float32), # Assuming joint limits
            "joint_velocity": spaces.Box(low=-20.0, high=20.0, shape=(12,), dtype=np.float32), # Assuming max velocity
            "base_position": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "base_orientation": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            "base_linear_velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "base_angular_velocity": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "base_linear_acceleration": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "feet_contact": spaces.MultiBinary(4),
            "lidar": spaces.Box(low=0.0, high=8.0, shape=(self.num_lidar_points,), dtype=np.float32),
            "camera": spaces.Box(low=0, high=255, shape=(self.expected_camera_height, self.expected_camera_width, 3), dtype=np.uint8)
        })


        # Define action space (12 joint positions)
        # Assuming joint position control within a reasonable range (e.g., -pi to pi)
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(self.num_joints,), dtype=np.float32)

        self._wait_for_initial_data()

    def _wait_for_initial_data(self):
        self.get_logger().info("Waiting for initial sensor data...")
        start_time = time.time()
        timeout = 30.0 # Increased timeout

        # Improved checks: verify that the arrays are populated with expected sizes
        # and that the camera image has been received.
        while not all([
            len(self.joint_positions) == 12,
            len(self.joint_velocities) == 12,
            len(self.joint_torques) == 12,
            len(self.base_position) == 3,
            len(self.base_orientation) == 4,
            len(self.base_linear_velocity) == 3,
            len(self.base_angular_velocity) == 3,
            len(self.base_linear_acceleration) == 3,
            len(self.feet_contact) == 4,
            len(self.lidar_ranges) == self.num_lidar_points,
            self.latest_camera_image is not None # Check if image has been received
        ]) and (time.time() - start_time < timeout):
            self.executor.spin_once(timeout_sec=0.001)
            time.sleep(0.01) # Small sleep to allow ROS 2 to process messages

        if (time.time() - start_time) >= timeout:
            self.get_logger().error("Timeout waiting for initial sensor data! Proceeding with potentially stale data.")
        else:
            self.get_logger().info("Initial sensor data received.")


    def _get_obs(self):
        # Combine all sensor data into a single observation dictionary
        obs = {
            "joint_state": self.joint_positions.astype(np.float32),
            "joint_velocity": self.joint_velocities.astype(np.float32),
            "base_position": self.base_position.astype(np.float32),
            "base_orientation": self.base_orientation.astype(np.float32),
            "base_linear_velocity": self.base_linear_velocity.astype(np.float32),
            "base_angular_velocity": self.base_angular_velocity.astype(np.float32),
            "base_linear_acceleration": self.base_linear_acceleration.astype(np.float32),
            "feet_contact": self.feet_contact.astype(np.int8), # MultiBinary expects int/bool
            "lidar": self.lidar_ranges.astype(np.float32),
            "camera": self.latest_camera_image if self.latest_camera_image is not None else np.zeros((self.expected_camera_height, self.expected_camera_width, 3), dtype=np.uint8)
        }
        return obs

    def _get_info(self):
        # You can add additional information here if needed for debugging or analysis
        return {"collided_with_obstacle": self.collided_with_obstacle}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.get_logger().info("Resetting simulation...") # Use self.get_logger()
        # Request simulation reset via ROS 2 service
        request = Empty.Request()
        future = self.reset_sim_client.call_async(request)

        # Spin the executor to allow the service call to be processed
        rclpy.spin_until_future_complete(self, future, executor=self.executor)

        if future.result() is not None:
            self.get_logger().info("Simulation reset complete.") # Use self.get_logger()
        else:
            self.get_logger().error("Failed to reset simulation.") # Use self.get_logger()

        # Reset internal state variables to their initial conditions (e.g., zeros, default values)
        # This is crucial because `_wait_for_initial_data` relies on fresh data.
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.joint_torques = np.zeros(12)
        self.base_position = np.zeros(3)
        self.base_orientation = np.zeros(4)
        self.base_linear_velocity = np.zeros(3)
        self.base_angular_velocity = np.zeros(3)
        self.base_linear_acceleration = np.zeros(3)
        self.feet_contact = np.zeros(4, dtype=bool)
        self.collided_with_obstacle = False
        self.lidar_ranges = np.full(360, 8.0)
        self.latest_camera_image = None

        # Wait for initial sensor data after reset
        self._wait_for_initial_data()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # Create and publish JointState message for joint commands
        joint_command_msg = JointState()
        joint_command_msg.header.stamp = self.get_clock().now().to_msg()
        joint_command_msg.position = action.tolist() # Convert numpy array to list
        self.joint_command_publisher.publish(joint_command_msg)

        # Spin ROS 2 to process incoming sensor data and send commands.
        # This loop processes `self.executor.spin_once` multiple times to ensure data is updated
        # and commands are sent/received within one simulation step.
        for _ in range(20): # Increased spin_once iterations for more robustness
            self.executor.spin_once(timeout_sec=0.001) # A small timeout for each spin
            # time.sleep(0.0005) # Optional: A very small sleep if CPU is being saturated

        observation = self._get_obs()
        reward = 0.0 # Define your reward function here
        terminated = False # Define termination conditions
        truncated = False # Define truncation conditions
        info = self._get_info()

        # Example termination condition: fall
        if observation["base_position"][2] < 0.15: # If robot falls below a certain height
            self.get_logger().info(f"Robot fell! Z-position: {observation['base_position'][2]}")
            terminated = True

        # Example reward: encourage forward movement, penalize joint effort, penalize collision
        reward = observation["base_linear_velocity"][0] * 0.1 # Forward velocity in X
        reward -= np.sum(np.square(action)) * 0.001 # Penalize large actions/effort
        if self.collided_with_obstacle:
            reward -= 10.0 # Large penalty for collision
            terminated = True # Terminate on collision

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Renders the environment.
        For 'rgb_array' mode, returns the latest camera image.
        """
        if self.render_mode == 'rgb_array':
            if self.latest_camera_image is not None:
                # Gymnasium expects an RGB image of shape (height, width, 3)
                # Ensure the image is the expected size
                if (self.latest_camera_image.shape[0] == self.expected_camera_height and
                    self.latest_camera_image.shape[1] == self.expected_camera_width and
                    self.latest_camera_image.shape[2] == 3):
                    return self.latest_camera_image
                else:
                    self.get_logger().warn( # Use self.get_logger()
                        f"Camera image shape mismatch. Expected ({self.expected_camera_height}, "
                        f"{self.expected_camera_width}, 3), got {self.latest_camera_image.shape}. "
                        "Returning black image."
                    )
                    return np.zeros((self.expected_camera_height, self.expected_camera_width, 3), dtype=np.uint8)
            else:
                self.get_logger().warn("No camera image received yet for rendering. Returning black image.") # Use self.get_logger()
                return np.zeros((self.expected_camera_height, self.expected_camera_width, 3), dtype=np.uint8)
        return None

    def close(self):
        self.executor.shutdown()
        self.destroy_node() # Destroy the ROS node

    # --- ROS 2 Callbacks ---
    def joint_state_callback(self, msg):
        self.joint_positions = np.array(msg.position)
        self.joint_velocities = np.array(msg.velocity)
        self.joint_torques = np.array(msg.effort)

    def imu_callback(self, msg):
        self.base_orientation = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.base_angular_velocity = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.base_linear_acceleration = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

    def odometry_callback(self, msg):
        self.base_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        self.base_linear_velocity = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])

    def contact_detection_callback(self, msg):
        self.feet_contact = np.array(msg.feet_stance, dtype=bool)
        self.collided_with_obstacle = msg.collided_with_obstacle

    def laser_scan_callback(self, msg):
        # Convert tuple to list then to numpy array
        self.lidar_ranges = np.array(list(msg.ranges), dtype=np.float32)
        # Replace inf with max_range if present, or handle NaN if any
        self.lidar_ranges[np.isinf(self.lidar_ranges)] = msg.range_max
        self.lidar_ranges[np.isnan(self.lidar_ranges)] = msg.range_max # Or a safe value

    def camera_image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self.latest_camera_image = np.array(cv_image)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}") # Use self.get_logger()