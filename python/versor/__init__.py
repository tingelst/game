''' Python interface for the Versor library '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.0.1"
__author__ = "Lars Tingelstad"
__email__ = "lars.tingelstad@ntnu.no"

import sys
sys.path.append('../../build/')

import numpy as np
from versor_pybind11 import *

Dls = Pnt

def to_array(self): 
    arr = np.zeros((4,4),dtype=np.float64)
    arr[:3,0] = np.array(Drv(1,0,0).spin(self))
    arr[:3,1] = np.array(Drv(0,1,0).spin(self))
    arr[:3,2] = np.array(Drv(0,0,1).spin(self))
    arr[:3,3] = np.array(Vec(0,0,0).null().spin(self))[:3]
    return arr

Mot.matrix = to_array


# from versor.cga import *
# from versor.utils import array
