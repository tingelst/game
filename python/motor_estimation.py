import sys

import numpy as np
np.set_printoptions(precision=6, suppress=True)

import sys
sys.path.append('../build_gcc/')

from libmotor_estimation import MotorEstimationSolver

import versor as vsr
from game import solver_options

options = solver_options()
options['minimizer_progress_to_stdout'] = True
options['parameter_tolerance'] = 1e-12
options['function_tolerance'] = 1e-12
options['max_num_iterations'] = 100
options['num_threads'] = 100
options['num_linear_solver_threads'] = 100

M_object_in_world = vsr.Trs.from_vector([1,1,0.1]) * vsr.Rot.from_bivector(vsr.Biv(1,0,0) * np.pi/4.0)
M_eye_in_hand = vsr.Trs.from_vector([0.1,0.2,0.3]) * vsr.Rot.from_bivector(vsr.Biv(0,1,0) * np.pi/6.)
M_eye_in_hand_initial = vsr.Trs.from_vector([0.11,0.24,0.36]) * vsr.Rot.from_bivector(vsr.Biv(1,1,0).unit() * np.pi/5.)

def hand_eye_pose_pairs(M_object_in_world, M_eye_in_hand, n):
    pose_pairs = np.array([(M_hand_in_world,
                            (M_eye_in_hand.rev() * M_hand_in_world.rev() * M_object_in_world))
                        for M_hand_in_world in [vsr.Trs.from_vector(vsr.Vec(*np.random.rand(3))) *
                                                vsr.Rot.from_bivector(vsr.Biv(*np.random.rand(3)).unit() *
                                                                        np.random.rand() * np.pi)
                                                for i in range(n)]])
    n = len(pose_pairs)
    As = pose_pairs[:,0]
    Bs = pose_pairs[:,1]
    LAs = []
    LBs = []
    for i in range(n):
        for j in range(i+1,n):
            LAs.append(((As[j].rev() * As[i]).log() * 0.5).unit())
            LBs.append(((Bs[j] * Bs[i].rev()).log() * 0.5).unit())
    return LAs, LBs

# motor = vsr.Mot.from_dir_ang_trs(vsr.Vec(0,1,0).unit(), np.pi/3, vsr.Vec(1,0,1))
def estimate_motor(cost_function_num, parameterization_num, num_elements, points_a=None, points_b_noisy=None):

    motor = vsr.Trs.from_vector(vsr.Vec(1,1,1)) * vsr.Rot.from_bivector(vsr.Biv(0,1,0) * np.pi/6.0)

    motor = M_eye_in_hand

    error_motor = vsr.Mot.from_dir_ang_trs([1,0,0], np.pi/12, [0,0,0])

    n = num_elements
    # Generate initial point sets
    if points_a is None and points_b_noisy is None:

        points_a = [vsr.Vec(*np.random.normal(0.0, 0.8, 3)).null() for i in range(n)]
        points_b = [point.spin(motor) for point in points_a]

        mu = 0.0
        sigma = 0.00
        add_noise_to_point = lambda p : (point.to_array()[:3] + sigma *  np.random.randn(1,3) + mu)[0]
        points_b_noisy = [vsr.Vec(*add_noise_to_point(point)).null()
                      for point in points_b ]

    lines_a = [(vsr.Vec(*np.random.normal(0.0, 0.8, 3)).null() ^
                (vsr.Vec(*np.random.normal(0.0, 0.8, 3)).unit() ^ vsr.ni)).dual()
            for i in range(n)]
    lines_b = [line.spin(motor) for line in lines_a]

    # initial_mot = motor.spin(error_motor)
    # initial_mot = vsr.Mot(1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    initial_mot = M_eye_in_hand_initial


    mes = MotorEstimationSolver(initial_mot, options)

    if cost_function_num == 1:
        print("game::Line Plucker line index difference. 6 residuals")
        for a, b in zip(lines_a, lines_b):
            mes.add_line_correspondences_residual_block(a, b)

    elif cost_function_num == 2:
        print("game::Line vector + angle. 4 residuals")
        for a, b in zip(lines_a, lines_b):
            mes.add_line_angle_distance_residual_block(a, b)

    elif cost_function_num == 3:
        print("game::Line distance + angle. 2 residuals")
        for a, b in zip(lines_a, lines_b):
            mes.add_line_angle_distance_norm_residual_block(a, b)

    elif cost_function_num == 4:
        print("game::Point distance (-0.5*d^2). 1 residuals")
        for a, b in zip(points_a, points_b_noisy):
            mes.add_point_distance_residual_block(a,b)

    elif cost_function_num == 5:
        print("game::Point vector. 3 residuals")
        for a, b in zip(points_a, points_b_noisy):
            mes.add_point_correspondences_residual_block(a,b)

    elif cost_function_num == 6:
        print("game:: ADEPT Point vector. 3 residuals")
        for a, b in zip(points_a, points_b_noisy):
            mes.add_adept_point_correspondences_residual_block(a,b)

    elif cost_function_num == 7:
        print("game::Point vector. 3 residuals")
        for a, b in zip(points_a, points_b_noisy):
            mes.add_point_correspondences_residual_block(a,b)
        print("game::Line vector + angle. 4 residuals")
        for a, b in zip(lines_a, lines_b):
            mes.add_line_angle_distance_residual_block(a, b)

    elif cost_function_num == 8:
        print("Hand Eye Calibration: Line angle distance")
        motor = M_eye_in_hand
        global LAs, LBs
        LAs, LBs = hand_eye_pose_pairs(M_object_in_world, M_eye_in_hand, n)
        for a, b in zip(LBs, LAs):
            mes.add_line_angle_distance_residual_block(a, b)

    elif cost_function_num == 9:
        print("Hand Eye Calibration: Correspondences")
        motor = M_eye_in_hand
        global LAs, LBs
        LAs, LBs = hand_eye_pose_pairs(M_object_in_world, M_eye_in_hand, n)
        for a, b in zip(LBs, LAs):
            mes.add_line_correspondences_residual_block(a, b)

    if parameterization_num == 1:
        mes.set_parameterization('POLAR_DECOMPOSITION')
    elif parameterization_num == 2:
        mes.set_parameterization('BIVECTOR_GENERATOR')
    elif parameterization_num == 3:
        mes.set_parameterization('BIVECTOR_GENERATOR_ADEPT')
    elif parameterization_num == -1:
        pass

    final_motor, summary = mes.solve()
    print("game:: Original motor")
    print(motor)
    print("game:: Estimated motor")
    print(final_motor)

    points_b_estimated = [point.spin(final_motor) for point in points_a]
    total_distance = np.sum([ np.linalg.norm(a.to_array()[:3] - b.to_array()[:3])
                           for a, b in zip(points_b, points_b_estimated)]) / np.sqrt(n)

    print("game:: Total RMS distance")
    print(total_distance)

    print(summary['brief_report'])

    return summary, final_motor

if __name__ == "__main__":
    estimate_motor(8,2,100)
