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

@ARTICLE{Park+Martin-1994,
  author =	 {Park, F.C. and Martin, B.J.},
  journal =	 {IEEE Transactions on Robotics and Automation},
  title =	 {{Robot Sensor Calibration: Solving AX = XB on the
                  Euclidean Group}},
  year =	 {1994},
  month =	 {oct},
  volume =	 {10},
  number =	 {5},
  pages =	 {717 -721},
  doi =		 {10.1109/70.326576},
  ISSN =	 {1042-296X},
}
"""

__author__ = "Morten Lind"
__credits__ = ["Morten Lind"]
__copyright__ = "SINTEF Raufoss Manufacturing AS 2011-2012"
__license__ = "GPLv3"
__maintainer__ = "Morten Lind"
__email__ = "morten.lind@sintef.no"
__status__ = "Development"

import numpy as np
import math3d as m3d


# class DirectoryReader(object):
#     """A reader of flange in base and object in sensor poses from
#     files matched in a given folder. The patterns given must specify a
#     group named 'n' which matches some number in the file name. This
#     number is used to correspond the files for object in sensor pose
#     with the flange in base pose to form a pair."""
#     def __init__(self, directory, flange_file_pattern, object_file_pattern):
#         self._ois_pattern = object_file_pattern
#         self._fib_pattern = flange_file_pattern
#         self._dir = directory

class _Move(object):
    """A move is a convenience class that computes and keeps a
    relative transform between two transforms. For convenience it
    stores the corresponding rotation as a rotation vector."""
    def __init__(self, trf_start, trf_end):
        self.move = trf_start.inverse * trf_end
        self.orient = self.move.orient
        self.d = self.move.pos
        self.rot = self.move.orient.rotation_vector.data


class ParkMartinCalibrator(object):

    class Exception(Exception):
        def __init__(self, msg):
            self._msg = msg
        def __repr__(sef):
            return 'ParkMartinCalibratorException : %s' % self._msg

    def __init__(self, pose_pairs=None, cond_cut = 10e-5):
        """The 'pose_pairs' is a sequence of, at least three, pairs
        (np.array, tuple or list of two) of flange to base and sensor
        to object math3d transforms. The object observed by the sensor
        is some object reference fixed with respect to the robot
        base. The sensor reference is to be given with respect to this
        fixed object reference."""
        self._cond_cut = cond_cut
        self.pose_pairs = pose_pairs

    def _compute_moves(self):
        """Compute the moves, i.e. the relative transforms, for the
        frame and sensor pairs."""
        move_pairs = []
        pp0 = self._pose_pairs[0]
        for pp in self._pose_pairs[1:]:
            move_pairs.append((_Move(pp0[0], pp[0]), _Move(pp0[1], pp[1])))
        self._move_pairs = np.array(move_pairs)

    @property
    def pose_pairs(self):
        return self._pose_pairs

    @pose_pairs.setter
    def pose_pairs(self, pose_pairs):
        """ Set a new set of flange and sensor pose pairs. Clear the
        stored sensor to flange transform."""
        self._pose_pairs = np.array([], dtype=np.object)
        self._pose_pairs.shape = (0, 2)
        self._move_pairs = np.array([], dtype=np.object)
        self._move_pairs.shape = (0, 2)
        self._invalidate()
        if not pose_pairs is None:
            self += pose_pairs

    def _invalidate(self):
        self._m = None
        self._c = None
        self._c_svd = None
        self._c_pinv = None
        self._mtm_svd = None
        self._mtm_sqrt_inv = None
        self._orient_sif = None
        self._pos_sif = None
        self._sensor_in_flange = None

    def __iadd__(self, pose_pairs):
        """ Add one or more pose pairs. The shape of the array in
        'pose_pairs' must be either (N,2) for adding N pose pairs, or
        (2,) for adding a single pose pair, and the array elements
        must be m3d.Transform. In both cases, the ordering of the
        pairs are flange to base and sensor to object transform."""
        if type(pose_pairs) != np.ndarray:
            pose_pairs = np.array(pose_pairs)
        pp_shape = pose_pairs.shape
        if pp_shape == (2,):
            # // Was given a list of one pose pair
            pose_pairs = np.array([pose_pairs])
        elif len(pp_shape) == 2 and pp_shape[1] == 2:
            # // Was given a sequence of pairs
            pass
        else:
            raise ParkMartinCalibrator.Exception('The given pose pair(s) was not well formated; needs (N,2) or (2,) np.array of Transforms.')
        if len(self._pose_pairs) == 0:
            self._pose_pairs = pose_pairs
        else:
            self._pose_pairs = np.vstack((self._pose_pairs, pose_pairs))
        if len(self._pose_pairs) > 1:
            move_pairs = []
            pp0 = self._pose_pairs[0]
            for pp in pose_pairs[1:]:
                move_pairs.append((_Move(pp0[0], pp[0]), _Move(pp0[1], pp[1])))
            if len(self._move_pairs) == 0:
                self._move_pairs = np.array(move_pairs)
            else:
                np.vstack(self._move_pairs, move_pairs)
        self._invalidate()
        return self

    @property
    def M(self):
        """ Compute, if necessary, the M matrix."""
        if self._m is None:
            self._m = np.sum([np.outer(sm.rot, fm.rot) for fm, sm in self._move_pairs], axis=0)
        return self._m

    @property
    def MTM_SVD(self):
        """ Compute, if necessary, the SVD of M^T *M."""
        if self._mtm_svd is None:
            self._mtm_svd = np.linalg.svd(self.M.T.dot(self.M), full_matrices=False)
        return self._mtm_svd

    @property
    def MTM_sqrt_inv(self):
        """ Compute, if necessary, (M^T * M)^(-1/2)."""
        if self._mtm_sqrt_inv is None:
            U, s, V = self.MTM_SVD
            self._mtm_sqrt_inv = U.dot(np.diag(1/np.sqrt(s))).dot(V)
        return self._mtm_sqrt_inv

    @property
    def C(self):
        """ Compute, if necessary, the C matrix."""
        # // C is the vertical stacking of I - Theta_A_i, the orientations of the flange moves
        if self._c is None:
            I_3 = np.identity(3)
            self._c = np.vstack([I_3 - fm.orient._data for fm,sm in self._move_pairs])
        return self._c

    @property
    def C_SVD(self):
        if self._c_svd is None:
            self._c_svd = np.linalg.svd(self.C, full_matrices=False)
        return self._c_svd

    @property
    def C_pinv(self):
        if self._c_pinv is None:
            U, s, V = self.C_SVD
            cutoff = s[0] * self._cond_cut
            s_recip = np.zeros(len(s))
            for i in range(len(s)):
                if s[i] > cutoff:
                    s_recip[i] = 1.0/s[i]
                else:
                    s_recip[i] = 0.0
            self._c_pinv = V.T.dot(np.diag(s_recip).dot(U.T))
        return self._c_pinv


    @property
    def orient_nai(self):
        """ Compute the noise amplification index of the M^T*M as the
        s_min^2/s_max. This gives good identifiability for kinematic
        calibration, according to Nahvi and Hollerbach (1996), but its
        usefulness as an index for the goodness of this particular
        matrix, and the orientational identification in
        hand-eye-calibration is unknown. mtm_nai should be larger for
        ensuring better orientation calibration"""
        s = self.MTM_SVD[1]
        return s[-1]**2/s[0]

    @property
    def orient_low_sing(self):
        return self.MTM_SVD[1][-1]

    @property
    def pos_nai(self):
        """ Compute the noise amplification index for the C
        matrix. See comment for mtm_nai."""
        s = self.C_SVD[1]
        return s[-1]**2/s[0]

    @property
    def pos_low_sing(self):
        return self.C_SVD[1][-1]

    @property
    def orient_sif(self):
        # // Compute the orientation of the solution, Theta_X, as
        # (Mt*M)^(-1/2)*Mt, and make the corresponding
        # orientation.
        if self._orient_sif is None:
            print(self.MTM_sqrt_inv.dot(self.M.T))
            self._orient_sif = m3d.Orientation(self.MTM_sqrt_inv.dot(self.M.T))
        return self._orient_sif

    @property
    def pos_sif(self):
        if self._pos_sif is None:
            # // d is the vertical stacking of b_A_i - Theta_X*b_B_i
            d = np.hstack([(fm.d- self.orient_sif*sm.d)._data for fm,sm in self._move_pairs])
            # // The solution to the position, b_X, is now computed as (C^T*C)^(-1)*C*d
            self._pos_sif =m3d.Vector(self.C_pinv.dot(d))
        return self._pos_sif

     # def _solve(self):
     #    """This protected method solves the Park Martin AX=XB
     #    calibration. It is assumed as a pre-condition that the flange
     #    sensor pairs have been set."""
     #    # // Using symbols from Park and Martin (1994). The move pairs
     #    # relate as follows: first entry of a move pair is flange
     #    # move, and its 'rot' member gives the orientation logarithm,
     #    # alpha. The second entry is the sensor move, and its 'rot'
     #    # member holds beta.
     #    # // M is the sum of outer products of betas with alphas: M = sum (beta*alpha^T)


    @property
    def sensor_in_flange(self):
        """Return the calibration result, i.e. the sensor to flange
        transform. If necessary, it is computed first"""
        if self._sensor_in_flange is None:
            if self._pose_pairs is None:
                raise Exception('The sensor in flange transform has not been'
                                +' computed yet, and flange-sensor pose pairs have'
                                + ' not been set!!!')
            else:
                self._sensor_in_flange = m3d.Transform(self.orient_sif, self.pos_sif)
        return self._sensor_in_flange

