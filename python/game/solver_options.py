from __future__ import absolute_import, division, print_function
def solver_options():
    solver_options = {
        'minimizer_type':'TRUST_REGION',
        'trust_region_strategy_type':'LEVENBERG_MARQUARDT',
        'linear_solver_type':'DENSE_QR',
        'max_num_iterations': 25,
        'num_threads': 4,
        'num_linear_solver_threads':4,
        'parameter_tolerance': 10e-8,
        'function_tolerance': 10e-8,
        'gradient_tolerance': 10e-12,
        'minimizer_progress_to_stdout':True,
        'trust_region_minimizer_iterations_to_dump':[],
        'trust_region_problem_dump_directory':'',
    }
    return solver_options
