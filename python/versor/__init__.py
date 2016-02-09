''' Python interface for the Versor library '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.0.1"
__author__ = "Lars Tingelstad"
__email__ = "lars.tingelstad@ntnu.no"

import sys
sys.path.append('../../build/')
# sys.path.append('../../build_gcc/')
# sys.path.append('../build/Debug/')
# sys.path.append('../../build/Debug/')


from versor.cga import *
from versor.utils import array
