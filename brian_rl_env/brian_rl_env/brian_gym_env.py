import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

# ROS 2 message imports
from sensor_msgs.msg import JointState, Imu, LaserScan, Image # Added Image
from brian_msgs.msg import ContactDetection
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

# OpenCV and CvBridge for image processing
import cv2 # Ensure you have opencv-python installed (pip install opencv-python)
from cv_bridge import CvBridge, CvBridgeError # Added CvBridge and CvBridgeError

class BrianGymEnv(gym.Env):
    """
    OpenAI Gym environment for the Brian quadruped robot in PyBullet,
    communicating via ROS 2 topics.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30} # Set render_fps for video speed

    def __init__(self, robot_name='brian', render_mode=None):
        # Initialize ROS 2 communication within the environment
        self.ros_node = Node('brian_gym_env_node')
        self.callback_group = ReentrantCallbackGroup()

        # Store the render_mode passed to the constructor
        self.render_mode = render_mode

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- ROS 2 Subscribers ---
        self.joint_state_sub = self.ros_node.create_subscription(
            JointState, f'/{robot_name}/joints_state', self._joint_state_callback, qos_profile,
            callback_group=self.callback_group
        )
        self.imu_sub = self.ros_node.create_subscription(
            Imu, f'/{robot_name}/imu_data', self._imu_callback, qos_profile,
            callback_group=self.callback_group
        )
        self.odom_sub = self.ros_node.create_subscription(
            Odometry, f'/{robot_name}/odom_data', self._odom_callback, qos_profile,
            callback_group=self.callback_group
        )
        self.contact_sub = self.ros_node.create_subscription(
            ContactDetection, f'/{robot_name}/contact_detection', self._contact_callback, qos_profile,
            callback_group=self.callback_group
        )
        self.lidar_sub = self.ros_node.create_subscription(
            LaserScan, '/scan', self._lidar_callback, qos_profile,
            callback_group=self.callback_group
        )

        # --- Camera Image Subscriber ---
        self.bridge = CvBridge() # Initialize CvBridge
        self.latest_camera_image = None # To store the latest camera frame as a NumPy array
        self.camera_sub = self.ros_node.create_subscription(
            Image, f'/{robot_name}/camera/image_raw', self._camera_callback, qos_profile,
            callback_group=self.callback_group
        )

        # --- ROS 2 Publisher ---
        self.joint_control_pub = self.ros_node.create_publisher(
            JointState, f'/{robot_name}/joints_control', 10
        )

        # --- State Variables (updated by callbacks) ---
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.base_orientation = np.zeros(4)
        self.base_angular_velocity = np.zeros(3)
        self.base_linear_acceleration = np.zeros(3)
        self.base_position = np.zeros(3)
        self.base_linear_velocity = np.zeros(3)
        self.feet_contact = np.zeros(4, dtype=bool)
        self.is_colliding_with_obstacle = False
        self.lidar_ranges = np.full(360, 8.0)

        # --- Gym Environment Setup ---
        self.joint_limits_lower = np.array([-1.571, 0.00000, -2.35619, # FL Hip1, Hip2, Knee
                                            -1.2010, 0.00000, -2.35619, # FR Hip1, Hip2, Knee
                                            -1.571, 0.00000, -2.35619, # BL Hip1, Hip2, Knee
                                            -1.2010, 0.00000, -2.35619]) # BR Hip1, Hip2, Knee
        self.joint_limits_upper = np.array([1.2010, 2.35619, -0.52360,
                                            1.571, 2.35619, -0.52360,
                                            1.2010, 2.35619, -0.52360,
                                            1.571, 2.35619, -0.52360])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # Define camera resolution expected from the PyBullet node
        self.expected_camera_height = 240
        self.expected_camera_width = 320

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(404,), dtype=np.float32
        )
        self.max_episode_steps = 500

        self.current_step = 0
        self.initial_base_x_pos = 0.0
        self.last_base_x_pos = 0.0

        # ROS 2 Service client for Reset
        self.reset_client = self.ros_node.create_client(Empty, '/brian/reset_sim')
        self.ros_node.get_logger().info("Waiting for /brian/reset_sim service...")
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.ros_node.get_logger().info('reset service not available, waiting again...')
        self.ros_node.get_logger().info("Reset service available.")

        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.ros_node)

        # Wait for initial data to populate all sensor variables
        self._wait_for_initial_data()

    def _wait_for_initial_data(self):
        self.ros_node.get_logger().info("Waiting for initial sensor data...")
        start_time = time.time()
        timeout = 10.0 # Increased timeout for initial data

        while not all([
            np.any(self.joint_positions != 0),
            np.any(self.base_position != 0),
            np.any(self.base_orientation != 0),
            np.any(self.lidar_ranges != 8.0),
            self.latest_camera_image is not None # Wait for the first camera image
        ]) and (time.time() - start_time < timeout):
            self.executor.spin_once(timeout_sec=0.001)
            time.sleep(0.01)

        if (time.time() - start_time) >= timeout:
            self.ros_node.get_logger().error("Timeout waiting for initial sensor data! Proceeding with potentially stale data.")
        else:
            self.ros_node.get_logger().info("Initial sensor data received.")


    # --- ROS 2 Callback Functions ---
    def _joint_state_callback(self, msg: JointState):
        self.joint_positions = np.array(msg.position)
        self.joint_velocities = np.array(msg.velocity)

    def _imu_callback(self, msg: Imu):
        self.base_orientation = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.base_angular_velocity = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.base_linear_acceleration = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

    def _odom_callback(self, msg: Odometry):
        self.base_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        self.base_linear_velocity = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])

    def _contact_callback(self, msg: ContactDetection):
        self.feet_contact = np.array(msg.feet_stance, dtype=bool)
        self.is_colliding_with_obstacle = msg.collided_with_obstacle

    def _lidar_callback(self, msg: LaserScan):
        self.lidar_ranges = np.array(msg.ranges)
        # Pad or truncate lidar ranges to 360 if necessary
        if len(self.lidar_ranges) != 360:
            padded_ranges = np.full(360, msg.range_max)
            copy_len = min(len(self.lidar_ranges), 360)
            padded_ranges[:copy_len] = self.lidar_ranges[:copy_len]
            self.lidar_ranges = padded_ranges

    def _camera_callback(self, msg: Image):
        try:
            # Convert ROS Image message to OpenCV image (NumPy array)
            # The encoding should match what your PyBullet node is publishing ('rgb8' or 'bgr8')
            # 'rgb8' is common for PyBullet.
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            # Ensure the image is C-contiguous for MoviePy/Gymnasium
            self.latest_camera_image = np.ascontiguousarray(cv_image)
        except CvBridgeError as e:
            self.ros_node.get_logger().error(f"CvBridge Error: {e}")
            self.latest_camera_image = None # Set to None on error

    # --- Gym Environment Methods ---
    def _get_obs(self):
        # Concatenate all observation components
        obs = np.concatenate([
            self.base_position,
            self._quat_to_euler(self.base_orientation), # Convert quaternion to Euler angles
            self.base_linear_velocity,
            self.base_angular_velocity,
            self.base_linear_acceleration,

            self.joint_positions,
            self.joint_velocities,

            self.feet_contact.astype(np.float32), # Convert boolean contacts to float
            np.array([float(self.is_colliding_with_obstacle)]), # Convert boolean collision to float

            self.lidar_ranges
        ]).astype(np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Call the ROS 2 reset service
        request = Empty.Request()
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self.ros_node, future, executor=self.executor)
        if future.result() is not None:
            self.ros_node.get_logger().info("PyBullet simulation reset via service.")
        else:
            self.ros_node.get_logger().error(f"Failed to call reset service: {future.exception()}. This could cause issues.")

        # After reset, ensure sensor data is fresh, including the camera image
        self._wait_for_initial_data()

        self.initial_base_x_pos = self.base_position[0]
        self.last_base_x_pos = self.initial_base_x_pos

        observation = self._get_obs()
        info = {}

        # If render_mode is set, call render once after reset for the video wrapper
        if self.render_mode == 'rgb_array':
            self.render() # Capture initial frame for video

        return observation, info

    def step(self, action):
        self.current_step += 1

        joint_cmd_msg = JointState()
        target_joint_positions = self._scale_action_to_joint_limits(action)
        joint_cmd_msg.position = target_joint_positions.tolist()
        self.joint_control_pub.publish(joint_cmd_msg)

        # Spin ROS 2 to process incoming sensor data and send commands.
        # This loop should be long enough to ensure PyBullet node has processed step and published data
        for _ in range(10): # Increased spin_once count for better data synchronization
            self.executor.spin_once(timeout_sec=0.001)
            # You might need to adjust this delay based on your simulation's publishing rate
            # time.sleep(0.001) # A small sleep can help if your PyBullet node is slower

        observation = self._get_obs()
        reward, terminated, collided_with_obstacle_flag = self._calculate_reward()

        truncated = self.current_step >= self.max_episode_steps

        info = {"collided_with_obstacle": collided_with_obstacle_flag}

        # If render_mode is set, call render to capture frames
        if self.render_mode == 'rgb_array':
            self.render()

        return observation, reward, terminated, truncated, info

    def _scale_action_to_joint_limits(self, action):
        scaled_action = self.joint_limits_lower + (0.5 * (action + 1.0) * (self.joint_limits_upper - self.joint_limits_lower))
        return scaled_action

    def _quat_to_euler(self, quat):
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(quat)
        return r.as_euler('xyz', degrees=False)

    def _calculate_reward(self):
        reward = 0.0
        terminated = False
        collided_with_obstacle_flag = False

        # --- Walking Reward ---
        forward_progress = self.base_position[0] - self.last_base_x_pos
        reward += forward_progress * 100.0
        self.last_base_x_pos = self.base_position[0]

        # Stability/Height
        roll, pitch, _ = self._quat_to_euler(self.base_orientation)

        current_base_z_pos = self.base_position[2]
        target_height = 0.25 # Adjust target height if necessary
        height_deviation = abs(current_base_z_pos - target_height)
        reward -= height_deviation * 50.0

        orientation_penalty = abs(roll) + abs(pitch)
        reward -= orientation_penalty * 20.0

        # Penalize large joint velocities
        joint_vel_penalty = np.sum(np.abs(self.joint_velocities))
        reward -= joint_vel_penalty * 0.01

        # Penalize falling
        if current_base_z_pos < 0.15: # Threshold for considering robot "fallen"
            reward -= 500.0
            terminated = True

        # --- Obstacle Avoidance ---
        if self.is_colliding_with_obstacle:
            reward -= 1000.0
            terminated = True
            collided_with_obstacle_flag = True

        min_lidar_distance = np.min(self.lidar_ranges)
        # Penalize getting too close to obstacles
        if min_lidar_distance < 0.5:
            proximity_penalty = 1.0 - (min_lidar_distance / 0.5) # Closer is higher penalty
            reward -= proximity_penalty * 100.0

        return reward, terminated, collided_with_obstacle_flag

    def render(self):
        """
        Returns the latest camera image as a NumPy array if render_mode is 'rgb_array'.
        Otherwise, returns None.
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
                    self.ros_node.get_logger().warn(
                        f"Camera image shape mismatch. Expected ({self.expected_camera_height}, "
                        f"{self.expected_camera_width}, 3), got {self.latest_camera_image.shape}. "
                        "Returning black image."
                    )
                    return np.zeros((self.expected_camera_height, self.expected_camera_width, 3), dtype=np.uint8)
            else:
                self.ros_node.get_logger().warn("No camera image received yet for rendering. Returning black image.")
                # Return a black image if no image is available yet to prevent errors
                return np.zeros((self.expected_camera_height, self.expected_camera_width, 3), dtype=np.uint8)
        return None # For other render modes, or if no rendering is desired

    def close(self):
        self.ros_node.get_logger().info("Shutting down BrianGymEnv ROS node.")
        self.executor.shutdown()
        self.ros_node.destroy_node()