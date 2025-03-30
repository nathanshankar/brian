#!/usr/bin/env -S python3

import pybullet as pb
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node


class simpleLog:
    def debug(self, msg):
        print('DEBUG: {:s}'.format(msg))

    def info(self, msg):
        print('INFO:  {:s}'.format(msg))

    def warn(self, msg):
        print('WARN:  {:s}'.format(msg))

    def error(self, msg):
        print('ERROR: {:s}'.format(msg))


class brianSim(Node):
    def __init__(self, urdf_dir, startPos, start_euler, ground, logging=None, rgba=[0.2, 0.2, 0.2, 1]):
        super().__init__('brian_sim')
        self.urdf_dir = urdf_dir
        self.startPos = startPos
        self.startOri = np.array(start_euler) * np.pi / 180
        self.ground = ground

        self.robot = pb.loadURDF(urdf_dir, self.startPos, pb.getQuaternionFromEuler(self.startOri))
        self.num_joints = pb.getNumJoints(self.robot)

        for i in range(-1, self.num_joints):
            pb.changeVisualShape(self.robot, i, rgbaColor=rgba)

        if logging is not None:
            self.logging = logging
        else:
            self.logging = simpleLog()

        self.prev_lin_vel = [0, 0, 0]
        self.prev_ang_vel = [0, 0, 0]

        self.timer = time.time()

        names = self.getJointNames()
        self.index_revolute_joints = []

        for i in range(len(names)):
            if 'hip' in names[i] or 'knee' in names[i]:
                self.index_revolute_joints.append(i)

    def getJointNames(self):
        joint_infos = [pb.getJointInfo(self.robot, i) for i in range(self.num_joints)]
        joint_names = [joint_infos[i][1].decode() for i in range(self.num_joints)]
        return joint_names

    def getRevoluteJointNames(self):
        joint_infos = [pb.getJointInfo(self.robot, i) for i in range(self.num_joints)]
        joint_names = [joint_infos[i][1].decode() for i in self.index_revolute_joints]
        return joint_names

    def getMotorJointStates(self):
        joint_state = [pb.getJointState(self.robot, i) for i in self.index_revolute_joints]
        joint_positions = [joint_state[i][0] for i in range(len(joint_state))]
        joint_velocities = [joint_state[i][1] for i in range(len(joint_state))]
        joint_torques = [joint_state[i][3] for i in range(len(joint_state))]
        return joint_positions, joint_velocities, joint_torques

    def getInfoLinks(self):
        info = [pb.getDynamicsInfo(self.robot, i) for i in range(-1, self.num_joints)]
        masses = [info[i][0] for i in range(len(info))]
        local_inertia = [info[i][2] for i in range(len(info))]
        return masses, local_inertia

    def getRobotState(self):
        lin_pos, ang_pos = pb.getBasePositionAndOrientation(self.robot)
        lin_vel, ang_vel = pb.getBaseVelocity(self.robot)

        dt = time.time() - self.timer
        self.timer = time.time()

        acc = np.true_divide(np.array(lin_vel) - np.array(self.prev_lin_vel), dt)
        ang_acc = np.true_divide(np.array(ang_vel) - np.array(self.prev_ang_vel), dt)

        self.prev_lin_vel = lin_vel
        self.prev_ang_vel = ang_vel

        return lin_pos, lin_vel, ang_pos, ang_vel, acc, ang_acc

    def getLidarData(self, max_dist, res_deg):
        lin_pos, ang_pos = pb.getBasePositionAndOrientation(self.robot)
        rot_matrix = R.from_quat(ang_pos).as_matrix()

        all_angles = np.arange(0, 2 * np.pi, res_deg * (np.pi / 180))

        final_pos = np.array([max_dist * np.cos(all_angles), max_dist * np.sin(all_angles),
                              (lin_pos[2] + 0.3) * np.ones(len(all_angles))])
        lidar_pos = np.array([lin_pos[0] * np.ones(len(all_angles)), lin_pos[1] * np.ones(len(all_angles)),
                              (lin_pos[2] + 0.3) * np.ones(len(all_angles))])

        final_pos_trans = np.dot(rot_matrix, final_pos)

        collisions = pb.rayTestBatch(lidar_pos.T, final_pos_trans.T)

        collision_pos = np.array([collision[3] for collision in collisions if collision[0] != -1])

        if collision_pos.size == 0:
            return all_angles, np.full(len(all_angles), max_dist)

        distance = np.sqrt(np.power(collision_pos[:, 0] - lin_pos[0], 2) +
                           np.power(collision_pos[:, 1] - lin_pos[1], 2) +
                           np.power(collision_pos[:, 2] - lin_pos[2], 2))

        return all_angles, distance

    def setJointPosition(self, joint_pos):
        if len(joint_pos) == len(self.index_revolute_joints):
            for i in range(len(self.index_revolute_joints)):
                joint_ix = self.index_revolute_joints[i]
                pb.setJointMotorControl2(self.robot, joint_ix, pb.POSITION_CONTROL, joint_pos[i], maxVelocity=20)
        else:
            self.get_logger().error(
                'Expected position array of length {}, got {}'.format(len(self.index_revolute_joints), len(joint_pos)))

    def getFeetContact(self):
        feet_contacts = np.zeros(4, dtype='bool')
        contacts = pb.getContactPoints(bodyA=self.ground, bodyB=self.robot)
        for contact in contacts:
            link_index = contact[4]
            if link_index >= 0:
                link_name = (pb.getJointInfo(self.robot, link_index)[12]).decode()
            else:
                link_name = 'base'

            if link_name == 'tibia_fl_link':
                feet_contacts[0] = 1
            if link_name == 'tibia_fr_link':
                feet_contacts[1] = 1
            if link_name == 'tibia_bl_link':
                feet_contacts[2] = 1
            if link_name == 'tibia_br_link':
                feet_contacts[3] = 1

        return feet_contacts
