# brian_gym_env.py (MODIFIED)
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

# ROS 2 message imports (ensure these are available)
from sensor_msgs.msg import JointState, Imu, LaserScan
from brian_msgs.msg import ContactDetection
from nav_msgs.msg import Odometry # Added for base position
from std_srvs.srv import Empty  # Import Empty service for reset

class BrianGymEnv(gym.Env):
    """
    OpenAI Gym environment for the Brian quadruped robot in PyBullet,
    communicating via ROS 2 topics.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, robot_name='brian', render_mode=None):
        # Initialize ROS 2 communication within the environment
        # DO NOT call rclpy.init() here. It's called once in the training script.
        self.ros_node = Node('brian_gym_env_node')
        self.callback_group = ReentrantCallbackGroup()

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
        self.odom_sub = self.ros_node.create_subscription( # NEW ODOM SUBSCRIPTION
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
        self.base_position = np.zeros(3) # NEW: For Odometry
        self.base_linear_velocity = np.zeros(3) # NEW: For Odometry
        self.feet_contact = np.zeros(4, dtype=bool)
        self.is_colliding_with_obstacle = False # NEW: For collision status
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

        # Observation Space size update
        # Base (pos, ori, ang_vel, lin_acc, lin_vel) - 3+4+3+3+3 = 16 (Using quaternion directly from IMU)
        # OR Base (pos, euler, ang_vel, lin_acc, lin_vel) - 3+3+3+3+3 = 15 (Converting to Euler)
        # Let's stick with Euler for simplicity as it's common in RL for stability.
        # Base (pos, euler, lin_vel, ang_vel, lin_acc) = 3 + 3 + 3 + 3 + 3 = 15
        # Joint states (pos, vel) - 12*2 = 24
        # Feet contacts - 4
        # Collision status - 1
        # Lidar ranges - 360
        # Total size: 15 + 24 + 4 + 1 + 360 = 404
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

        # Wait for initial data to populate
        self._wait_for_initial_data()


    def _wait_for_initial_data(self):
        self.ros_node.get_logger().info("Waiting for initial sensor data...")
        start_time = time.time()
        timeout = 10.0 # seconds
        
        while not all([
            np.any(self.joint_positions != 0),
            np.any(self.base_position != 0), # Check base position from Odometry
            np.any(self.base_orientation != 0),
            np.any(self.lidar_ranges != 8.0)
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

    def _odom_callback(self, msg: Odometry): # NEW ODOM CALLBACK
        self.base_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        self.base_linear_velocity = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])

    def _contact_callback(self, msg: ContactDetection):
        self.feet_contact = np.array(msg.feet_stance, dtype=bool)
        self.is_colliding_with_obstacle = msg.collided_with_obstacle # NEW: Get collision status

    def _lidar_callback(self, msg: LaserScan):
        self.lidar_ranges = np.array(msg.ranges)
        if len(self.lidar_ranges) != 360:
            padded_ranges = np.full(360, msg.range_max)
            copy_len = min(len(self.lidar_ranges), 360)
            padded_ranges[:copy_len] = self.lidar_ranges[:copy_len]
            self.lidar_ranges = padded_ranges


    # --- Gym Environment Methods ---
    def _get_obs(self):
        obs = np.concatenate([
            self.base_position, # (3)
            self._quat_to_euler(self.base_orientation), # (3)
            self.base_linear_velocity, # (3)
            self.base_angular_velocity, # (3)
            self.base_linear_acceleration, # (3)
            
            self.joint_positions, # (12)
            self.joint_velocities, # (12)
            
            self.feet_contact.astype(np.float32), # (4)
            np.array([float(self.is_colliding_with_obstacle)]), # (1)
            
            self.lidar_ranges # (360)
        ]).astype(np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Call the ROS 2 reset service
        request = Empty.Request()
        future = self.reset_client.call_async(request)
        # Spin the executor to allow the service call to complete
        rclpy.spin_until_future_complete(self.ros_node, future, executor=self.executor)
        if future.result() is not None:
            self.ros_node.get_logger().info("PyBullet simulation reset via service.")
        else:
            self.ros_node.get_logger().error(f"Failed to call reset service: {future.exception()}. This could cause issues.")
            # Depending on severity, you might want to raise an exception here.

        # After reset, ensure sensor data is fresh
        self._wait_for_initial_data()
        
        self.initial_base_x_pos = self.base_position[0] # Get current X from freshly updated Odom
        self.last_base_x_pos = self.initial_base_x_pos
        
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        self.current_step += 1

        joint_cmd_msg = JointState()
        target_joint_positions = self._scale_action_to_joint_limits(action)
        joint_cmd_msg.position = target_joint_positions.tolist()
        self.joint_control_pub.publish(joint_cmd_msg)

        # Spin ROS 2 to process incoming sensor data and send commands.
        # It's crucial to spin enough times to let the PyBullet node respond.
        # Consider the time step of PyBullet (0.001s) and your mainThread timer.
        # A few spin_onces is usually enough.
        for _ in range(5): # Spin a few times to allow sensor data to update
            self.executor.spin_once(timeout_sec=0.001)
            
        observation = self._get_obs()
        reward, terminated, collided_with_obstacle_flag = self._calculate_reward() # Use different name for flag

        truncated = self.current_step >= self.max_episode_steps

        info = {"collided_with_obstacle": collided_with_obstacle_flag}
        return observation, reward, terminated, truncated, info

    def _scale_action_to_joint_limits(self, action):
        scaled_action = self.joint_limits_lower + (0.5 * (action + 1.0) * (self.joint_limits_upper - self.joint_limits_lower))
        return scaled_action

    def _quat_to_euler(self, quat):
        from scipy.spatial.transform import Rotation as R # Import here to avoid circular dependency with brian_sim if it imports this
        r = R.from_quat(quat)
        return r.as_euler('xyz', degrees=False)

    def _calculate_reward(self):
        reward = 0.0
        terminated = False
        collided_with_obstacle_flag = False # New flag to return in info

        # --- Walking Reward ---
        forward_progress = self.base_position[0] - self.last_base_x_pos
        reward += forward_progress * 100.0
        self.last_base_x_pos = self.base_position[0]

        # Stability/Height
        roll, pitch, _ = self._quat_to_euler(self.base_orientation)
        
        current_base_z_pos = self.base_position[2]
        target_height = 0.25 # Tune this
        height_deviation = abs(current_base_z_pos - target_height)
        reward -= height_deviation * 50.0

        orientation_penalty = abs(roll) + abs(pitch)
        reward -= orientation_penalty * 20.0

        # Penalize large joint velocities
        joint_vel_penalty = np.sum(np.abs(self.joint_velocities))
        reward -= joint_vel_penalty * 0.01

        # Penalize falling
        if current_base_z_pos < 0.15:
            reward -= 500.0
            terminated = True

        # --- Obstacle Avoidance ---
        if self.is_colliding_with_obstacle:
            reward -= 1000.0 # Very large penalty for collision
            terminated = True
            collided_with_obstacle_flag = True

        min_lidar_distance = np.min(self.lidar_ranges)
        if min_lidar_distance < 0.5:
            proximity_penalty = 1.0 - (min_lidar_distance / 0.5)
            reward -= proximity_penalty * 100.0
            
        return reward, terminated, collided_with_obstacle_flag

    def render(self):
        pass

    def close(self):
        self.ros_node.get_logger().info("Shutting down BrianGymEnv ROS node.")
        self.executor.shutdown()
        self.ros_node.destroy_node()
        # DO NOT call rclpy.shutdown() here. It's called once in the training script.