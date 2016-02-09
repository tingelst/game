#ifndef GAME_GAME_CERES_PYTHON_UTILS_H_
#define GAME_GAME_CERES_PYTHON_UTILS_H_

#include <pybind11/pybind11.h>
#include <ceres/ceres.h>
#include <glog/logging.h>

namespace py = pybind11;

namespace game {

  /*
void CheckContiguousArray(const np::ndarray& m, const std::string& name) {
  if (!(m.get_flags() & np::ndarray::C_CONTIGUOUS)) {
    std::stringstream ss;
    ss << name << ": Array is not c contiguous";
    PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
    bp::throw_error_already_set();
  }
}

void CheckCols(const np::ndarray& m, const std::string& name, int cols) {
  int cols_actual = m.shape(1);
  if (!(cols_actual == cols)) {
    std::stringstream ss;
    ss << name << ": Array must have shape (N," << cols << ")";
    PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
    bp::throw_error_already_set();
  }
}
void CheckRows(const np::ndarray& m, const std::string& name, int rows) {
  int rows_actual = m.shape(0);
  if (!(rows_actual == rows)) {
    std::stringstream ss;
    ss << name << ": Array must have shape (" << rows << ", n)";
    PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
    bp::throw_error_already_set();
  }
}

void CheckArrayShape(const np::ndarray& m, const std::string& name, int rows,
                     int cols) {
  int rows_actual = m.shape(0);
  int cols_actual = m.shape(1);
  if (!((rows_actual == rows) && (cols_actual == cols))) {
    std::stringstream ss;
    ss << name << ": Array must have shape (" << rows << "," << cols << ")";
    PyErr_SetString(PyExc_RuntimeError, ss.str().c_str());
    bp::throw_error_already_set();
  }
}

void CheckContiguousArrayAndArrayShape(const np::ndarray& m,
                                       const std::string& name, int rows,
                                       int cols) {
  CheckContiguousArray(m, name);
  CheckArrayShape(m, name, rows, cols);
}

  */

auto SummaryToDict(const ceres::Solver::Summary& summary) -> py::dict {
  py::dict summary_dict;
  summary_dict[py::str("linear_solver_type_used")] = py::str(
      ceres::LinearSolverTypeToString(summary.linear_solver_type_used));

  summary_dict[py::str("num_parameter_blocks")] = py::int_(summary.num_parameter_blocks);
  summary_dict[py::str("num_parameters")] = py::int_(summary.num_parameters);
  summary_dict[py::str("num_residual_blocks")] = py::int_(summary.num_residual_blocks);
  summary_dict[py::str("num_residuals")] = py::int_(summary.num_residuals);
  summary_dict[py::str("trust_region_strategy_type")] = py::str(ceres::TrustRegionStrategyTypeToString(summary.trust_region_strategy_type));
  summary_dict[py::str("minimizer_type")] =
    py::str(ceres::MinimizerTypeToString(summary.minimizer_type));

  py::list iterations;
  auto its = summary.iterations;
  for (int i = 0; i < its.size(); ++i) {
    py::dict iteration;
    iteration[py::str("iteration")] = py::int_(its[i].iteration);
    iteration[py::str("cost")] = py::float_(its[i].cost);
    iteration[py::str("cost_change")] = py::float_(its[i].cost_change);
    iteration[py::str("gradient_max_norm")] = py::float_(its[i].gradient_max_norm);
    iteration[py::str("step_norm")] = py::float_(its[i].step_norm);
    iteration[py::str("relative_decrease")] = py::float_(its[i].relative_decrease);
    iteration[py::str("trust_region_radius")] = py::float_(its[i].trust_region_radius);
    iteration[py::str("eta")] = py::float_(its[i].eta);
    iteration[py::str("linear_solver_iterations")] = py::int_(its[i].linear_solver_iterations);
    iterations.append(iteration);
  }
  summary_dict[py::str("iterations")] = iterations;
  summary_dict[py::str("brief_report")] = py::str(summary.BriefReport().c_str());
  summary_dict[py::str("full_report")] = py::str(summary.FullReport().c_str());
  return summary_dict;
}

/*
void SetSolverOptions(const bp::dict& solver_options,
                      ceres::Solver::Options& options) {
  bp::extract<std::string> linear_solver_type(
      solver_options["linear_solver_type"]);
  if (linear_solver_type.check()) {
    ceres::StringToLinearSolverType(linear_solver_type(),
                                    &options.linear_solver_type);
  }

  bp::extract<int> max_num_iterations(solver_options["max_num_iterations"]);
  if (max_num_iterations.check()) {
    options.max_num_iterations = max_num_iterations();
  }

  bp::extract<int> num_threads(solver_options["num_threads"]);
  if (num_threads.check()) {
    options.num_threads = num_threads();
  }

  bp::extract<int> num_linear_solver_threads(
      solver_options["num_linear_solver_threads"]);
  if (num_linear_solver_threads.check()) {
    options.num_linear_solver_threads = num_linear_solver_threads();
  }

  bp::extract<double> parameter_tolerance(
      solver_options["parameter_tolerance"]);
  if (parameter_tolerance.check()) {
    options.parameter_tolerance = parameter_tolerance();
  }

  bp::extract<double> function_tolerance(solver_options["function_tolerance"]);
  if (function_tolerance.check()) {
    options.function_tolerance = function_tolerance();
  }

  bp::extract<double> gradient_tolerance(solver_options["gradient_tolerance"]);
  if (gradient_tolerance.check()) {
    options.gradient_tolerance = gradient_tolerance();
  }

  bp::extract<std::string> trust_region_strategy_type(
      solver_options["trust_region_strategy_type"]);
  if (trust_region_strategy_type.check()) {
    ceres::StringToTrustRegionStrategyType(trust_region_strategy_type(),
                                           &options.trust_region_strategy_type);
  }

  bp::extract<bool> minimizer_progress_to_stdout(
      solver_options["minimizer_progress_to_stdout"]);
  if (minimizer_progress_to_stdout.check()) {
    options.minimizer_progress_to_stdout = minimizer_progress_to_stdout();
  }

  bp::extract<std::string> minimizer_type(solver_options["minimizer_type"]);
  if (minimizer_type.check()) {
    ceres::StringToMinimizerType(minimizer_type(), &options.minimizer_type);
  }

  bp::extract<bp::list> trust_region_minimizer_iterations_to_dump(
      solver_options["trust_region_minimizer_iterations_to_dump"]);
  if (trust_region_minimizer_iterations_to_dump.check()) {
    std::vector<int> iterations_to_dump{};
    bp::list list = trust_region_minimizer_iterations_to_dump();
    for (int i = 0; i < bp::len(list); ++i) {
      iterations_to_dump.push_back(bp::extract<int>(list[i])());
    }
    options.trust_region_minimizer_iterations_to_dump = iterations_to_dump;
  }

  bp::extract<std::string> trust_region_problem_dump_directory(
      solver_options["trust_region_problem_dump_directory"]);
  if (trust_region_problem_dump_directory.check()) {
    options.trust_region_problem_dump_directory =
        trust_region_problem_dump_directory();
  }
}
*/

}
#endif  // GAME_GAME_CERES_PYTHON_UTILS_H_
