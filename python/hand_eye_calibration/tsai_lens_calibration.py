# -*- coding: utf-8 -*-
"""
A general module for solving the eye-in-hand calibration where a the
analysis results of a vision system refers to some unknown coordinate
reference, e.g. the calibration reference for the vision system, fixed
with a controlled coordinate reference, e.g. a tool flange of a
robot. It is the objective of the eye-in-hand calibration to determine
the (fixed) transformation between the controlled coordinate reference
and the vision system coordinate reference.

The method in this module uses an external calibration object, fixed
with respect to the base reference for the controlled coordinate
reference, that can be fully localized by the vision system. By moving
the controlled coordinate reference to a recorded set of poses, while
querying the vision analysis and recording the corresponding object
poses, a full identification of the vision system coordinate reference
can be performed.

The specific method implemented in this module is based on the one
described in

@ARTICLE{Tsai+Lenz-1989,
  author={Tsai, R.Y. and Lenz, R.K.},
  journal={Robotics and Automation, IEEE Transactions on},
  title={A new technique for fully autonomous and efficient 3D robotics
            hand/eye calibration},
  year={1989},
  volume={5},
  number={3},
  pages={345-358},
  doi={10.1109/70.34770},
  ISSN={1042-296X},
}

This implementation is based directly on the Matlab scripts by
Zoran Lazarevic found at:
    http://lazax.com/www.cs.columbia.edu/~laza/html/Stewart/matlab/

"""

__author__ = "Lars Tingelstad"
__credits__ = ["Lars Tingelstad"]
__copyright__ = "NTNU 2013"
__license__ = "GPLv3"
__maintainer__ = "Lars Tingelstad"
__email__ = "lars.tingelstad@ntnu.no"
__status__ = "Development"

import numpy as np
import math3d as m3d


class TsaiLenzCalibrator(object):

    def __init__(self, pose_pairs=None):
        self.pose_pairs = pose_pairs

    def _solve(self):

        # // Calculate rotational component

        pp = self.pose_pairs
        M = len(pp)
        lhs = []
        rhs = []
        for i in range(M):
            for j in range(i+1,M):
                Hgij = pp[j][0].inverse * pp[i][0]
                Pgij = 2 * Hgij.orient.quaternion.vector_part
                Hcij = pp[j][1].inverse * pp[i][1]
                Pcij = 2 * Hcij.orient.quaternion.vector_part
                lhs.append(skew(Pgij.array + Pcij.array))
                rhs.append(Pcij.array - Pgij.array)
        lhs = np.array(lhs)
        lhs = lhs.reshape(lhs.shape[0]*3, 3)
        rhs = np.array(rhs)
        rhs = rhs.reshape(rhs.shape[0]*3)
        Pcg_, res, rank, sing = np.linalg.lstsq(lhs, rhs)
        Pcg = 2 * Pcg_ / np.sqrt(1 + np.dot(Pcg_.reshape(3), Pcg_.reshape(3)))
        Rcg = quat_to_rot(Pcg / 2)

        # // Calculate translational component
        lhs = []
        rhs = []
        for i in range(M):
            for j in range(i+1,M):
                Hgij = pp[j][0].inverse * pp[i][0]
                Hcij = pp[j][1].inverse * pp[i][1]
                lhs.append(Hgij.array[:3,:3] - np.eye(3))
                rhs.append(np.dot(Rcg[:3,:3], Hcij.pos.array) - Hgij.pos.array)
        lhs = np.array(lhs)
        lhs = lhs.reshape(lhs.shape[0]*3, 3)
        rhs = np.array(rhs)
        rhs = rhs.reshape(rhs.shape[0]*3)
        Tcg, res, rank, sing = np.linalg.lstsq(lhs, rhs)
        Hcg = m3d.Transform(np.ravel(Rcg[:3,:3]), Tcg)
        self._sif = Hcg

    @property
    def sensor_in_flange(self):
        self._solve()
        return self._sif

def skew(v):
    if len(v) == 4: v = v[:3]/v[3]
    skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
    return skv - skv.T

def quat_to_rot(q):
    q = np.array(q).reshape(3,1)
    p = np.dot(q.T,q).reshape(1)[0]
    if p > 1:
        print('Warning: quaternion greater than 1')
    w = np.sqrt(1 - p)
    R = np.eye(4)
    R[:3,:3] = 2*q*q.T + 2*w*skew(q) + np.eye(3) - 2*np.diag([p,p,p])
    return R

def rot_to_quat(r):
    w4 = 2 * np.sqrt(1 - np.trace(r[:3,:3]))
    q = np.array([(r[2,1] - r[1,2]) / w4,
                  (r[0,2] - r[2,0]) / w4,
                  (r[1,0] - r[0,1]) / w4])
    return q


if __name__ == '__main__':
    pp = np.load('eye2hand_test_pose_pairs.npy')
    tlc = TsaiLenzCalibrator(pp)
    sif_tsai = tlc.sensor_in_flange
    print(sif_tsai)

