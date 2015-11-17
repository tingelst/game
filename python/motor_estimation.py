import sys
sys.path.append('../build/Debug')

import numpy as np
np.set_printoptions(precision=6, suppress=True)

from libmotor_estimation import MotorEstimationSolver

import versor as vsr
from game import solver_options

motor = vsr.Mot.from_dir_ang_trs(vsr.Vec(3,2,1).unit(), np.pi/3, vsr.Vec(1,2,3))
error_motor = vsr.Mot.from_dir_ang_trs([1,0,0], np.pi/12, [0,0,0])


# Generate initial point sets
n = 10
points_a = [vsr.Vec(*np.random.normal(0.0, 0.8, 3)).null() for i in range(n)]
points_b = [point.spin(motor) for point in points_a]

lines_a = [(vsr.Vec(*np.random.normal(0.0, 0.8, 3)).null() ^
            (vsr.Vec(*np.random.normal(0.0, 0.8, 3)).unit() ^ vsr.ni)).dual()
           for i in range(n)]
lines_b = [line.spin(motor) for line in lines_a]

initial_mot = motor.spin(error_motor)
# initial_mot = vsr.Mot(1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)

options = solver_options()
# options['minimizer_progress_to_stdout'] = True
options['parameter_tolerance'] = 1e-12
options['function_tolerance'] = 1e-12

mes = MotorEstimationSolver(initial_mot, options)

for a, b in zip(lines_a, lines_b):
    mes.add_line_correspondences_residual_block(a, b)

for a, b in zip(points_a, points_b):
    mes.add_point_correspondences_residual_block(a,b)

final_motor, summary = mes.solve()
print(motor)
print(final_motor)
print(np.allclose(motor.to_array(), final_motor.to_array()))

points_b_estimated = [point.spin(final_motor) for point in points_a]
total_distance = np.sum([ np.linalg.norm(a.to_array()[:3] - b.to_array()[:3])
                          for a, b in zip(points_b, points_b_estimated)]) / np.sqrt(n)
print(total_distance)
                       

