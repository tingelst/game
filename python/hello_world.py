import sys
sys.path.append('../build/Debug')
import numpy as np

import libhello_world as hello_world

x = np.array([0.4])
op = hello_world.OptimizationProblem()
op.run(x)

summary = op.summary()
print(summary['iterations'])
