import pybullet as pb
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
# Note: In a pure PyBullet training environment, rclpy.node.Node might not be strictly necessary
# for brianSim itself unless it's designed to be a ROS node independently.
# For simplicity in this RL training setup, we can adapt it or ensure rclpy.init() is called.
# The previous fix ensured rclpy.init() is called, so keeping Node inheritance is fine.
from rclpy.node import Node


class simpleLog:
    """A simple logging class for PyBullet simulation, mimicking ROS 2 logger behavior."""
    def debug(self, msg):
        print(f'DEBUG: {msg}')

    def info(self, msg):
        print(f'INFO:  {msg}')

    def warn(self, msg):
        print(f'WARN:  {msg}')

    def error(self, msg):
        print(f'ERROR: {msg}')


class brianSim(Node):
    """
    Simulation class for the Brian quadruped robot in PyBullet.
    Handles robot loading, state retrieval, joint control, and sensor data.
    """
    def __init__(self, urdf_dir, startPos, start_euler, ground, logging=None, rgba=[0.2, 0.2, 0.2, 1], physicsClient=None):
        """
        Initializes the Brian robot simulation.

        Args:
            urdf_dir (str): Path to the URDF file of the robot.
            startPos (list): Initial [x, y, z] position of the robot.
            start_euler (list): Initial [roll, pitch, yaw] orientation in degrees.
            ground (int): PyBullet ID of the ground plane.
            logging (object, optional): A logging object. Defaults to simpleLog.
            rgba (list, optional): RGBA color for the robot. Defaults to [0.2, 0.2, 0.2, 1].
            physicsClient (int, optional): The PyBullet physics client ID. Defaults to None.
        """
        super().__init__('brian_sim') # Initialize as a ROS 2 node

        self.urdf_dir = urdf_dir
        self.startPos = startPos
        self.startOri = np.array(start_euler) * np.pi / 180 # Convert degrees to radians
        self.ground = ground
        self.physicsClient = physicsClient # Store the physics client ID

        # Load the URDF model using the specified physics client
        if self.physicsClient is not None:
            self.robot = pb.loadURDF(urdf_dir, self.startPos, pb.getQuaternionFromEuler(self.startOri), physicsClientId=self.physicsClient)
        else:
            # Fallback if no physicsClient is provided (e.g., for standalone brianSim)
            self.robot = pb.loadURDF(urdf_dir, self.startPos, pb.getQuaternionFromEuler(self.startOri))

        self.num_joints = pb.getNumJoints(self.robot)

        # Change visual properties of robot links
        for i in range(-1, self.num_joints): # -1 refers to the base link
            if self.physicsClient is not None:
                pb.changeVisualShape(self.robot, i, rgbaColor=rgba, physicsClientId=self.physicsClient)
            else:
                pb.changeVisualShape(self.robot, i, rgbaColor=rgba)

        if logging is not None:
            self.logging = logging
        else:
            self.logging = simpleLog() # Use the simple logger if no external one is provided

        self.prev_lin_vel = [0, 0, 0]
        self.prev_ang_vel = [0, 0, 0]
        self.timer = time.time()

        # Identify revolute joints (hip and knee) for control
        names = self.getJointNames()
        self.index_revolute_joints = []
        for i in range(len(names)):
            # Assuming 'hip' and 'knee' are in the names of revolute joints you want to control
            if 'hip' in names[i] or 'knee' in names[i]:
                self.index_revolute_joints.append(i)

    def _load_robot(self):
        robot_id = pb.loadURDF(self.urdf_dir, self.startPos, pb.getQuaternionFromEuler(self.startOri))
        return robot_id

    def _colorize_robot(self):
        for i in range(-1, pb.getNumJoints(self.robot)):
            pb.changeVisualShape(self.robot, i, rgbaColor=self.rgba)

    def reset_robot(self):
        self.logging.info("Removing old robot and loading new one...")
        if self.robot is not None:
            try:
                pb.removeBody(self.robot)
                self.logging.info(f"Successfully removed old robot (ID: {self.robot}).")
            except pb.error as e:
                self.logging.error(f"Error removing old robot: {e}")

        self.robot = self._load_robot()
        self.num_joints = pb.getNumJoints(self.robot)
        self._colorize_robot()

        self.prev_lin_vel = [0, 0, 0]
        self.prev_ang_vel = [0, 0, 0]
        self.timer = time.time()
        self.logging.info(f"New robot loaded (ID: {self.robot}).")

    def getJointNames(self):
        """Returns a list of all joint names in the robot."""
        joint_infos = [pb.getJointInfo(self.robot, i, physicsClientId=self.physicsClient) for i in range(self.num_joints)]
        joint_names = [joint_infos[i][1].decode() for i in range(self.num_joints)]
        return joint_names

    def getRevoluteJointNames(self):
        """Returns a list of names for the revolute joints identified for control."""
        joint_infos = [pb.getJointInfo(self.robot, i, physicsClientId=self.physicsClient) for i in range(self.num_joints)]
        joint_names = [joint_infos[i][1].decode() for i in self.index_revolute_joints]
        return joint_names

    def getMotorJointStates(self):
        """
        Returns the current position, velocity, and torque of the controlled revolute joints.
        """
        joint_state = [pb.getJointState(self.robot, i, physicsClientId=self.physicsClient) for i in self.index_revolute_joints]
        joint_positions = [state[0] for state in joint_state]
        joint_velocities = [state[1] for state in joint_state]
        joint_torques = [state[3] for state in joint_state] # Joint motor torque
        return joint_positions, joint_velocities, joint_torques

    def getInfoLinks(self):
        """
        Returns mass and local inertia of all links (including base).
        (Note: This function might not be directly used in the RL loop but is kept for completeness)
        """
        info = [pb.getDynamicsInfo(self.robot, i, physicsClientId=self.physicsClient) for i in range(-1, self.num_joints)]
        masses = [info[i][0] for i in range(len(info))]
        local_inertia = [info[i][2] for i in range(len(info))]
        return masses, local_inertia

    def getRobotState(self):
        """
        Returns the current state of the robot's base link.
        Includes linear position, linear velocity, angular position (quaternion),
        angular velocity, linear acceleration, and angular acceleration.
        """
        lin_pos, ang_pos_quat = pb.getBasePositionAndOrientation(self.robot, physicsClientId=self.physicsClient)
        lin_vel, ang_vel_rad = pb.getBaseVelocity(self.robot, physicsClientId=self.physicsClient)

        dt = time.time() - self.timer
        self.timer = time.time()

        # Calculate acceleration (simple finite difference)
        acc = np.true_divide(np.array(lin_vel) - np.array(self.prev_lin_vel), dt)
        ang_acc_rad = np.true_divide(np.array(ang_vel_rad) - np.array(self.prev_ang_vel), dt)

        self.prev_lin_vel = lin_vel
        self.prev_ang_vel = ang_vel_rad

        return lin_pos, lin_vel, ang_pos_quat, ang_vel_rad, acc, ang_acc_rad

    def getLidarData(self, max_dist, res_deg):
        """
        Simulates Lidar data for the robot.
        (Note: This might be computationally intensive for high-frequency RL.
        Consider simplifying or omitting if not critical for the task.)
        """
        lin_pos, ang_pos_quat = pb.getBasePositionAndOrientation(self.robot, physicsClientId=self.physicsClient)
        rot_matrix = R.from_quat(ang_pos_quat).as_matrix()

        # Generate angles for lidar rays
        all_angles = np.arange(0, 2 * np.pi, res_deg * (np.pi / 180))

        # Calculate start and end points for ray tests
        # Lidar is mounted on base_link at z + 0.08341, so adjust ray start/end points
        lidar_height = lin_pos[2] + 0.08341 # Approximate height of the lidar sensor
        
        # Start points of rays (from lidar position relative to robot base)
        start_x = lin_pos[0] + rot_matrix[0, 0] * 0.04312 + rot_matrix[0, 2] * 0.08341
        start_y = lin_pos[1] + rot_matrix[1, 0] * 0.04312 + rot_matrix[1, 2] * 0.08341
        start_z = lidar_height

        # End points of rays in world coordinates
        end_x = start_x + max_dist * np.cos(all_angles) * rot_matrix[0,0] + max_dist * np.sin(all_angles) * rot_matrix[0,1]
        end_y = start_y + max_dist * np.cos(all_angles) * rot_matrix[1,0] + max_dist * np.sin(all_angles) * rot_matrix[1,1]
        end_z = start_z + max_dist * np.cos(all_angles) * rot_matrix[2,0] + max_dist * np.sin(all_angles) * rot_matrix[2,1]

        # PyBullet rayTestBatch expects lists of [startX, startY, startZ] and [endX, endY, endZ]
        ray_from_points = [[start_x, start_y, start_z] for _ in all_angles]
        ray_to_points = np.column_stack((end_x, end_y, end_z)).tolist()


        collisions = pb.rayTestBatch(ray_from_points, ray_to_points, physicsClientId=self.physicsClient)

        distances = []
        for i, collision in enumerate(collisions):
            hit_object_uid, _, _, hit_position, _ = collision
            if hit_object_uid != -1:
                # Calculate distance if a collision occurred
                dist = np.linalg.norm(np.array(hit_position) - np.array(ray_from_points[i]))
                distances.append(dist)
            else:
                # No collision, maximum distance
                distances.append(max_dist)

        return all_angles, distances

    def setJointPosition(self, joint_pos, velocity=None): # Added velocity parameter
        """
        Sets the target position for the revolute joints.

        Args:
            joint_pos (list): Target joint positions for the revolute joints.
            velocity (float, optional): Maximum velocity for joint control. Defaults to None.
        """
        if len(joint_pos) == len(self.index_revolute_joints):
            for i in range(len(self.index_revolute_joints)):
                joint_ix = self.index_revolute_joints[i]
                if velocity is not None:
                    pb.setJointMotorControl2(self.robot, joint_ix, pb.POSITION_CONTROL, joint_pos[i],
                                             maxVelocity=velocity, physicsClientId=self.physicsClient) # Use velocity and physicsClient
                else:
                    pb.setJointMotorControl2(self.robot, joint_ix, pb.POSITION_CONTROL, joint_pos[i],
                                             physicsClientId=self.physicsClient) # Use physicsClient
        else:
            self.logging.error(
                f'Expected position array of length {len(self.index_revolute_joints)}, got {len(joint_pos)}')

    def getFeetContact(self):
        """
        Checks for contact points between the robot's feet and the ground.
        Returns a boolean array indicating contact for each foot (FL, FR, BL, BR).
        """
        feet_contacts = np.zeros(4, dtype='bool')
        contacts = pb.getContactPoints(bodyA=self.ground, bodyB=self.robot, physicsClientId=self.physicsClient)
        for contact in contacts:
            link_index = contact[4] # Link index of the robot that made contact
            if link_index >= 0:
                link_name = (pb.getJointInfo(self.robot, link_index, physicsClientId=self.physicsClient)[12]).decode()
            else:
                link_name = 'base' # Base link contact

            # Map link names to foot contact array indices
            if link_name == 'tibia_fl_link':
                feet_contacts[0] = True
            elif link_name == 'tibia_fr_link':
                feet_contacts[1] = True
            elif link_name == 'tibia_bl_link':
                feet_contacts[2] = True
            elif link_name == 'tibia_br_link':
                feet_contacts[3] = True

        return feet_contacts

