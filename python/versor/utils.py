from __future__ import absolute_import, division, print_function
import numpy as np

def array(multivector):
    return np.array([multivector[i] for i in range(multivector.num)])
