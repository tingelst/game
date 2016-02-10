from __future__ import absolute_import, division, print_function
def solver_options():
    solver_options = {
        u'minimizer_type': u'TRUST_REGION',
        u'trust_region_strategy_type': u'LEVENBERG_MARQUARDT',
        u'linear_solver_type': u'DENSE_QR',
        u'max_num_iterations': 25,
        u'num_threads': 4,
        u'num_linear_solver_threads':4,
        u'parameter_tolerance': 10e-8,
        u'function_tolerance': 10e-8,
        u'gradient_tolerance': 10e-12,
        u'minimizer_progress_to_stdout':True,
        u'trust_region_minimizer_iterations_to_dump':[],
        u'trust_region_problem_dump_directory':'',
    }
    return solver_options
