from __future__ import absolute_import, division, print_function
import numpy as np

from libversor import (Biv, Bst, Cir, Con, Dil, Dll, Dlp, Dls, Drb, Drt, Drv,
                       Flp, Grt, Inf, Lin, Mnk, Mot, Ori, Par, Pln, Pnt, Pss, Rot,
                       Sca, Sph, Tnb, Tnt, Tnv, Tri, Trs, Trv, Tsd, Vec)

_types = [Biv, Bst, Cir, Con, Dil, Dll, Dlp, Dls, Drb, Drt, Drv, Flp,
          Grt, Inf, Lin, Mnk, Mot, Ori, Par, Pln, Pnt, Pss, Rot,
          Sca, Sph, Tnb, Tnt, Tnv, Tri, Trs, Trv, Tsd, Vec]


def _to_array(self):
    return np.array([self[i] for i in range(self.num)])

# Inject methods to all multivector types
for t in _types:
    setattr(t, 'to_array', _to_array)

# Inject factory method into Trs
def _from_vector(self, vector):
    return Trs(1.0, -0.5 * vector[0], -0.5 * vector[1], -0.5 * vector[2])
setattr(Trs, 'from_vector', classmethod(_from_vector))

# Inject factory method into Rot
def _rotor_from_bivector_gen(self, bivector):
    th = bivector.norm() / 2
    b = bivector.unit()
    return Rot(np.cos(th), -np.sin(th) * b[0], -np.sin(th) * b[1], -np.sin(th) * b[2])
setattr(Rot, 'from_bivector', classmethod(_rotor_from_bivector_gen))

no = Ori(1.0)
ni = Inf(1.0)
I = Pss(1.0)
E = Mnk(1.0)

def _motor_from_dir_ang_trs(self, dir, ang, trs):
    d = dir
    a = ang / 2.0
    t = trs
    B23 =  d[0]
    B13 = -d[1]
    B12 =  d[2]
    cp = np.cos(a)
    sp = np.sin(a)
    t1 = t[0] / 2.0
    t2 = t[1] / 2.0
    t3 = t[2] / 2.0
    m = [cp,
         -B12 * sp,
         -B13 * sp,
         -B23 * sp,
         -t1 * cp + (-B12 * t2 - B13 * t3) * sp,
         -t2 * cp + ( B12 * t1 - B23 * t3) * sp,
         -t3 * cp + (-B13 * t1 + B23 * t2) * sp,
         (B12 * t3 - B13 * t2 + B23 * t1) * sp]
    return Mot(*m)
setattr(Mot, 'from_dir_ang_trs', classmethod(_motor_from_dir_ang_trs))
