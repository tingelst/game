import sys
sys.path.append('../')
import numpy as np
import versor as vsr
from motor_estimation_valkenburg_dorst import (point_matrix, dual_line_matrix,
                                               dual_plane_matrix)


class VDMotorEstimationSolver(object):
    def __init__(self):
        self._point_matrix = np.zeros((8, 8))
        self._dual_line_matrix = np.zeros((8, 8))
        self._dual_plane_matrix = np.zeros((8, 8))

    @property
    def L(self):
        return self._point_matrix +\
            self._dual_line_matrix +\
            self._dual_plane_matrix

    def add_point_observations(self, ps, qs):
        for p, q in zip(ps, qs):
            self._point_matrix += point_matrix(p, q)

    def add_dual_line_observations(self, ps, qs):
        for p, q in zip(ps, qs):
            self._dual_line_matrix += dual_line_matrix(p, q)

    def add_dual_plane_observations(self, ps, qs):
        for p, q in zip(ps, qs):
            self._dual_plane_matrix += dual_plane_matrix(p, q)

    def solve(self):
        L = self.L
        Lrr = L[:4, :4]
        Lrq = L[:4, 4:]
        Lqr = L[4:, :4]
        Lqq = L[4:, 4:]
        Lp = Lrr - np.dot(Lrq, np.dot(np.linalg.pinv(Lqq), Lqr))
        w, v = np.linalg.eig(Lp)
        r = v[:, np.argmax(w)]
        q = np.dot(-np.dot(np.linalg.pinv(Lqq), Lqr), r)
        return vsr.Mot(*np.array([r, q]).ravel())
