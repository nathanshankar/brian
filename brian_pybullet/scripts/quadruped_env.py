import gymnasium as gym
from gymnasium import spaces
import pybullet as pb
import pybullet_data
import numpy as np
import time
from brian_sim import brianSim

class QuadrupedEnv(gym.Env):
    """
    Custom Gym environment for a quadruped robot in PyBullet.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(self, urdf_path, robot_name="brian", render_mode=None):
        """
        Initializes the Quadruped Reinforcement Learning environment.

        Args:
            urdf_path (str): The path to the URDF file of the quadruped robot.
            robot_name (str): The name of the robot. Defaults to "brian".
            render_mode (str, optional): The rendering mode ('human' for GUI, 'rgb_array' for offscreen). Defaults to None (direct mode).
        """
        super(QuadrupedEnv, self).__init__()

        self.urdf_path = urdf_path
        self.robot_name = robot_name
        self.render_mode = render_mode

        # Initialize PyBullet
        if self.render_mode == 'human':
            self.client = pb.connect(pb.GUI)
        else:
            self.client = pb.connect(pb.DIRECT)

        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        pb.setGravity(0, 0, -9.8, physicsClientId=self.client)
        pb.setTimeStep(1/240, physicsClientId=self.client) # Simulation time step

        # Load ground plane
        self.ground = pb.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=self.client)
        pb.changeDynamics(self.ground, -1, lateralFriction=1, spinningFriction=0.01, rollingFriction=0.01, physicsClientId=self.client)

        # Robot initial position and orientation
        self.start_pos = [0, 0, 0.26]
        self.start_ori = [0, 0, 0] # Euler angles in degrees

        # Instantiate brianSim, passing the physicsClient
        self.robot = brianSim(self.urdf_path, self.start_pos, self.start_ori, self.ground,
                              logging=None, rgba=[0.2, 0.2, 0.2, 1], physicsClient=self.client)

        self.joint_names = self.robot.getRevoluteJointNames()
        self.num_joints = len(self.joint_names)

        # Define action space (normalized joint position commands for 12 revolute joints)
        # These bounds are based on your URDF and should be accurate.
        # Order: hip1_fl, hip1_fr, hip1_bl, hip1_br, hip2_fl, hip2_fr, hip2_bl, hip2_br, knee_fl, knee_fr, knee_bl, knee_br
        self.joint_lower_bounds = np.array([-1.571, -1.201, -1.571, -1.201, # Hip1 (around X-axis)
                                            0.0, 0.0, 0.0, 0.0,             # Hip2 (around Y-axis)
                                            -2.356, -2.356, -2.356, -2.356], dtype=np.float32) # Knee (around Y-axis)
        self.joint_upper_bounds = np.array([1.201, 1.571, 1.201, 1.571,  # Hip1
                                            2.356, 2.356, 2.356, 2.356,  # Hip2
                                            -0.523, -0.523, -0.523, -0.523], dtype=np.float32) # Knee

        # Action space for normalized values between -1 and 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32)

        # Define observation space
        # Base position (3), orientation (4-quat), linear velocity (3), angular velocity (3),
        # Joint positions (12), joint velocities (12), feet contact (4)
        # Total: 3 + 4 + 3 + 3 + 12 + 12 + 4 = 41
        obs_low = np.array([-np.inf]*3 + [-1.0]*4 + [-np.inf]*3 + [-np.inf]*3 + # Base state
                           list(self.joint_lower_bounds) + # Joint positions
                           [-np.inf]*self.num_joints + # Joint velocities
                           [0.0]*4, dtype=np.float32) # Feet contact (0 or 1)

        obs_high = np.array([np.inf]*3 + [1.0]*4 + [np.inf]*3 + [np.inf]*3 + # Base state
                            list(self.joint_upper_bounds) + # Joint positions
                            [np.inf]*self.num_joints + # Joint velocities
                            [1.0]*4, dtype=np.float32) # Feet contact (0 or 1)

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.prev_base_pos = np.zeros(3)
        self.initial_base_z = self.start_pos[2] # Initial z position of the robot

    def _normalize_action(self, action):
        """
        Maps normalized actions [-1, 1] to joint position commands within their limits.
        """
        # Linear interpolation from [-1, 1] to [lower_bound, upper_bound]
        return self.joint_lower_bounds + (0.5 * (action + 1.0) * (self.joint_upper_bounds - self.joint_lower_bounds))

    def _get_observation(self):
        """
        Returns the current observation of the robot.
        Concatenates base state, joint states, and feet contact.
        """
        lin_pos, lin_vel, ang_pos_quat, ang_vel_rad, acc, ang_acc_rad = self.robot.getRobotState()
        joint_positions, joint_velocities, _ = self.robot.getMotorJointStates()
        feet_contact = self.robot.getFeetContact()

        observation = np.concatenate([
            np.array(lin_pos, dtype=np.float32),
            np.array(ang_pos_quat, dtype=np.float32),
            np.array(lin_vel, dtype=np.float32),
            np.array(ang_vel_rad, dtype=np.float32),
            np.array(joint_positions, dtype=np.float32),
            np.array(joint_velocities, dtype=np.float32),
            np.array(feet_contact, dtype=np.float32)
        ]).flatten() # Ensure the observation is a 1D array
        return observation

    def step(self, action):
        """
        Applies an action to the robot and steps the simulation.
        Calculates reward and checks for termination conditions.

        Args:
            action (np.ndarray): The action to apply (normalized joint positions).

        Returns:
            tuple: observation, reward, terminated, truncated, info
        """
        # Initialize flags and info dictionary to ensure they are always present
        terminated = False
        truncated = False
        info = {}
        
        # Map normalized action to actual joint angles
        target_joint_positions = self._normalize_action(action)
        self.robot.setJointPosition(target_joint_positions, velocity=20)

        # Step the PyBullet simulation
        pb.stepSimulation(physicsClientId=self.client)

        observation = self._get_observation()
        
        # --- Reward Calculation ---
        lin_pos, lin_vel, ang_pos_quat, ang_vel_rad, _, _ = self.robot.getRobotState()
        current_x_pos = lin_pos[0]
        current_z_pos = lin_pos[2]

        # Reward for forward movement (along the X-axis)
        reward_forward = (current_x_pos - self.prev_base_pos[0]) * 100.0
        self.prev_base_pos = np.array(lin_pos)

        # Penalize large roll/pitch to encourage balance
        roll, pitch, yaw = pb.getEulerFromQuaternion(ang_pos_quat)
        reward_balance = - (abs(roll) + abs(pitch)) * 5.0 # Higher penalty for larger angles

        # Penalize if robot falls below a certain height
        reward_fall = 0.0 # Initialize to 0.0
        if current_z_pos < self.initial_base_z * 0.7: # If robot falls below 70% of initial height
            reward_fall = -50.0
            terminated = True
        
        # Encourage feet contact (small reward for feet on the ground)
        feet_contact = self.robot.getFeetContact()
        reward_contact = np.sum(feet_contact) * 0.1

        # Total reward
        reward = reward_forward + reward_balance + reward_fall + reward_contact

        # Additional termination conditions based on orientation
        if abs(roll) > 0.5 or abs(pitch) > 0.5: # If roll or pitch is too extreme (approx 28 degrees)
            terminated = True
            reward -= 20.0 # Additional penalty for extreme orientation

        # Return all 5 expected values: observation, reward, terminated, truncated, info
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.
        """
        super().reset(seed=seed) # Important for Gymnasium compatibility

        # Reset PyBullet simulation
        pb.resetSimulation(physicsClientId=self.client)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        pb.setGravity(0, 0, -9.8, physicsClientId=self.client)
        pb.setTimeStep(1/240, physicsClientId=self.client)

        # Reload the ground plane
        self.ground = pb.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=self.client)
        pb.changeDynamics(self.ground, -1, lateralFriction=1, spinningFriction=0.01, rollingFriction=0.01, physicsClientId=self.client)

        # Reload the robot, ensuring physicsClient is passed
        self.robot = brianSim(self.urdf_path, self.start_pos, self.start_ori, self.ground,
                              logging=None, rgba=[0.2, 0.2, 0.2, 1], physicsClient=self.client)

        self.prev_base_pos = np.array(self.start_pos) # Reset previous position for reward calculation

        observation = self._get_observation()
        info = {}
        return observation, info

    def render(self):
        """Renders the environment."""
        if self.render_mode == 'human':
            # PyBullet GUI handles rendering automatically in GUI mode
            time.sleep(1/self.metadata['render_fps']) # Slow down simulation for human viewing
            pass
        elif self.render_mode == 'rgb_array':
            # Get the robot's current position to make the camera follow the robot
            robot_lin_pos, _ = pb.getBasePositionAndOrientation(self.robot.robot, physicsClientId=self.client)

            view_matrix = pb.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[robot_lin_pos[0], robot_lin_pos[1], robot_lin_pos[2]], # Target is the robot's center
                distance=0.8,                     # Distance from target
                yaw=45,                           # Horizontal rotation (view from side-front)
                pitch=-20,                        # Vertical rotation (view from slightly above)
                roll=0,
                upAxisIndex=2,
                physicsClientId=self.client
            )
            proj_matrix = pb.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(960)/720,
                nearVal=0.1,
                farVal=100.0,
                physicsClientId=self.client
            )
            (_, _, px, _, _) = pb.getCameraImage(
                width=960,
                height=720,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=pb.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self.client
            )
            rgb_array = np.array(px, dtype=np.uint8).reshape((720, 960, 4))
            return rgb_array[:, :, :3] # Return RGB channels only
        
    def close(self):
        """Closes the PyBullet simulation."""
        pb.disconnect(physicsClientId=self.client)

