#!/usr/bin/env python
"""This module provides an forward kinematics solver and a tool velocity solver
for the Universal Robot based on Orocos
"""
from math import pi
import numpy as np
# import math3d as m3d
import PyKDL as kdl

__author__ = "Johannes Schrimpf"
__copyright__ = "Copyright 2012, NTNU"
__credits__ = ["Johannes Schrimpf"]
__license__ = "GPL"
__maintainer__ = "Johannes Schrimpf"
__email__ = "johannes.schrimpf(_at_)itk.ntnu.no"
__status__ = "Development"


# def m3d_transform_to_kdl_frame(transform):
    # """Converts math3d transformation to a kdl frame.

    # Arguments:
    # transform -- The math3d Transform
    # """
    # if isinstance(transform, m3d.Transform):
        # rot = kdl.Rotation.Quaternion(transform.orient.quaternion[1],
                                  # transform.orient.quaternion[2],
                                  # transform.orient.quaternion[3],
                                  # transform.orient.quaternion[0])
        # pos = kdl.Vector(*transform.pos)
        # return kdl.Frame(rot, pos)
    # else:
        # raise Exception("Conversion not defined")


# def kdl_frame_to_m3d_transform(frame):
    # """Converts a kdl frame to a math3d transformation.

    # Arguments:
    # frame -- The frame
    # """
    # if isinstance(frame, kdl.Frame):
        # quat = frame.M.GetQuaternion()
        # pos = frame.p
        # m3d_quat = m3d.Quaternion(quat[3], quat[0], quat[1], quat[2], norm_warn=False)
        # m3d_vector = m3d.Vector(pos[0], pos[1], pos[2])
        # transform = m3d.Transform(m3d_quat.orientation, m3d_vector)
        # return transform
    # else:
        # raise Exception("Conversion not defined for type %s" % type(frame))


class Kinematics:
    """Kinematics object that holds the forward and tool velocity solver.
    It has to be initialized with a list of segments."""
    def __init__(self, chain):
        """Initialize the Kinematics

        Arguments:
        chain -- PyKDL chain for the robot
        """
        self._num_of_joints = int(chain.getNrOfJoints())
        self._fk_solver_pos = None
        self._tv_solver = None
        self._ik_solver_pos = None

        self._base_chain = chain
        self._tool_segment = None
        self._complete_chain = None
        self._set_solvers()

    @property
    def num_of_joints(self):
        return self._num_of_joints

    def _set_solvers(self):
        """Initialize the solvers according to the stored base chain and
        tool"""
        self._complete_chain = kdl.Chain(self._base_chain)
        if self._tool_segment is not None:
            if isinstance(self._tool_segment, kdl.Segment):
                self._complete_chain.addSegment(self._tool_segment)
            else:
                raise Exception("Tool is not None, Chain or Segment")
        self._fk_solver_pos = kdl.ChainFkSolverPos_recursive(
                                                  self._complete_chain)
        self._tv_solver = kdl.ChainIkSolverVel_pinv(self._complete_chain)
        self._ik_solver_pos = kdl.ChainIkSolverPos_NR(self._complete_chain,
                                                  self._fk_solver_pos,
                                                  self._tv_solver)

    def set_tool_transform(self, tool=None):
        """Set a tool transformation

        Keyword Arguments
        tool -- The tool as either None, Segment, Frame or
                math3d Transform (default None)
        """
        if tool is None:
            self._tool_segment = None
        elif isinstance(tool, kdl.Segment):
            self._tool_segment = kdl.Segment(tool)
        elif isinstance(tool, kdl.Frame):
            self._tool_segment = kdl.Segment(kdl.Joint(), tool)
        elif isinstance(tool, m3d.Transform):
            frame = m3d_transform_to_kdl_frame(tool)
            self._tool_segment = kdl.Segment(kdl.Joint(), frame)
        else:
            raise Exception("Tool is not None, Segment, Frame or m3d Transform")
        self._set_solvers()
        return True

    def get_tool_transform(self):
        """Returns the tool transformation"""
        if self._tool_segment is None:
            return m3d.Transform()
        else:
            frame = self._tool_segment.getFrameToTip()
            return kdl_frame_to_m3d_transform(frame)

    def get_frame(self, q, frame_num=-1):
        """Returns the frame for a given joint configuration.

        Arguments:
        q -- joint configuration

        Keyword arguments:
        frame_num -- The number of the frame. Default returns
                     the tool frame (default -1)
        """
        if type(q) == 'JntArray':
            joint_arr = q
        else:
            joint_arr = kdl.JntArray(self._num_of_joints)
            for joint_num in range(self._num_of_joints):
                joint_arr[joint_num] = q[joint_num]
        frame = kdl.Frame()
        self._fk_solver_pos.JntToCart(joint_arr, frame, frame_num)
        return frame

    def get_ik_from_pose(self, pose, q):
        """Returns inverse kinematics for a given tool center frame

        Arguments:
        pose -- Tool center frame as m3d transform
        q -- joint angles
        """
        q_out = kdl.JntArray(self._num_of_joints)
        q_init = kdl.JntArray(self._num_of_joints)
        for joint in range(self._num_of_joints):
            q_init[joint] = q[joint]
        p_in = m3d_transform_to_kdl_frame(pose)
        if self._ik_solver_pos.CartToJnt(q_init, p_in, q_out) != 0:
            raise Exception('Could not solve IK for given pose: ' + str(pose))
        q_out_np = np.zeros(self._num_of_joints)
        for joint in range(self._num_of_joints):
            q_out_np[joint] = q_out[joint]
        return q_out_np

    def get_tool_pose(self, q):
        """Returns the tool center frame for a given joint configuration.

        Arguments:
        q -- joint configuration
        """
        joint_arr = kdl.JntArray(self._num_of_joints)
        for joint_num in range(self._num_of_joints):
            joint_arr[joint_num] = q[joint_num]
        frame = kdl.Frame()
        self._fk_solver_pos.JntToCart(joint_arr, frame, self._complete_chain.getNrOfSegments())
        return kdl_frame_to_m3d_transform(frame)

    def get_flange_pose(self, q):
        """Returns the flange pose for a given joint configuration.

        Arguments:
        q -- joint configuration
        """
        joint_arr = kdl.JntArray(self._num_of_joints)
        for joint_num in range(self._num_of_joints):
            joint_arr[joint_num] = q[joint_num]
        frame = kdl.Frame()
        if self._tool_segment is None:
            self._fk_solver_pos.JntToCart(joint_arr, frame, self._complete_chain.getNrOfSegments())
        else:
            self._fk_solver_pos.JntToCart(joint_arr, frame, self._complete_chain.getNrOfSegments() - 1)
        return kdl_frame_to_m3d_transform(frame)


    def print_all_frames(self, q):
        """Prints all frames for a given joint configuration.

        Arguments:
        q -- joint configuration
        """
        for frame_num in range(self._num_of_joints + 1):
            print(self.get_frame(q, frame_num))

    def get_qdot_from_twist(self, q, twist):
        """Returns the joint velocities for a given twist

        Arguments:
        q -- The current robot configuration
        twist -- The desired twist
        """
        q_arr = kdl.JntArray(self._num_of_joints)
        qdot_arr = kdl.JntArray(self._num_of_joints)
        kdl_twist = kdl.Twist()
        for joint_num in range(self._num_of_joints):
            q_arr[joint_num] = q[joint_num]
        for joint_num in range(6):
            kdl_twist[joint_num] = twist[joint_num]
        self._tv_solver.CartToJnt(q_arr, kdl_twist, qdot_arr)
        ret = np.array([0.0] * self._num_of_joints)
        for joint_num in range(self._num_of_joints):
            ret[joint_num] = qdot_arr[joint_num]
        return ret


class URKinematics(Kinematics):
    """This class represents an Universal Robot."""
    DH_A = [0, -0.425, -0.39243, 0, 0, 0]
    DH_D = [0.0892, 0, 0, 0.109, 0.093, 0.082]
    DH_ALPHA = [pi / 2, 0, 0, pi / 2, -pi / 2, 0]
    DH_Q_HOME_OFFSET = [0, -pi / 2, 0, -pi / 2, 0, 0]
    NUM_JOINTS = 6

    def __init__(self):
        """Initialize the Universal Robot class using its dh parmeters."""
        chain = kdl.Chain()
        for segment in range(self.NUM_JOINTS):
            joint = kdl.Joint(kdl.Joint.RotZ)
            frame = kdl.Frame().DH(self.DH_A[segment],
                               self.DH_ALPHA[segment],
                               self.DH_D[segment],
                               0)
            chain.addSegment(kdl.Segment(joint, frame))
        Kinematics.__init__(self, chain)


class TrackURKinematics(Kinematics):
    """This class represents an Universal Robot on a linear track ."""
    DH_A = [0, -0.425, -0.39243, 0, 0, 0]
    DH_D = [0.0892, 0, 0, 0.109, 0.093, 0.082]
    DH_ALPHA = [pi / 2, 0, 0, pi / 2, -pi / 2, 0]
    DH_Q_HOME_OFFSET = [0, -pi / 2, 0, -pi / 2, 0, 0]
    NUM_JOINTS = 6

    def __init__(self, linear_transform):
        """Initialize the Universal Robot class using its dh parmeters."""
        linear_frame = m3d_transform_to_kdl_frame(linear_transform)
        chain = kdl.Chain()
        # Add linear segment
        chain.addSegment(kdl.Segment(kdl.Joint(kdl.Joint.TransX), linear_frame))
        # Add UR segments
        for segment in range(self.NUM_JOINTS):
            joint = kdl.Joint(kdl.Joint.RotZ)
            frame = kdl.Frame().DH(self.DH_A[segment],
                               self.DH_ALPHA[segment],
                               self.DH_D[segment],
                               0)
            chain.addSegment(kdl.Segment(joint, frame))
        Kinematics.__init__(self, chain)

if __name__ == "__main__":
    ur = URKinematics()
