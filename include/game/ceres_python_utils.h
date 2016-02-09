#ifndef GAME_GAME_CERES_PYTHON_UTILS_H_
#define GAME_GAME_CERES_PYTHON_UTILS_H_

/* #include <boost/numpy.hpp> */
#include <boost/python.hpp>
#include <ceres/ceres.h>
#include <glog/logging.h>

namespace bp = boost::python;
/* namespace np = boost::numpy; */

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

auto SummaryToDict(const ceres::Solver::Summary& summary) -> bp::dict {
  bp::dict summary_dict;
  summary_dict["linear_solver_type_used"] = std::string(
      ceres::LinearSolverTypeToString(summary.linear_solver_type_used));
  summary_dict["num_parameter_blocks"] = summary.num_parameter_blocks;
  summary_dict["num_parameters"] = summary.num_parameters;
  summary_dict["num_residual_blocks"] = summary.num_residual_blocks;
  summary_dict["num_residuals"] = summary.num_residuals;
  summary_dict["trust_region_strategy_type"] =
      std::string(ceres::TrustRegionStrategyTypeToString(
          summary.trust_region_strategy_type));
  summary_dict["minimizer_type"] =
      std::string(ceres::MinimizerTypeToString(summary.minimizer_type));

  bp::list iterations;
  auto its = summary.iterations;
  for (int i = 0; i < its.size(); ++i) {
    bp::dict iteration;
    iteration["iteration"] = its[i].iteration;
    iteration["cost"] = its[i].cost;
    iteration["cost_change"] = its[i].cost_change;
    iteration["gradient_max_norm"] = its[i].gradient_max_norm;
    iteration["step_norm"] = its[i].step_norm;
    iteration["relative_decrease"] = its[i].relative_decrease;
    iteration["trust_region_radius"] = its[i].trust_region_radius;
    iteration["eta"] = its[i].eta;
    iteration["linear_solver_iterations"] = its[i].linear_solver_iterations;
    iterations.append(iteration);
  }
  summary_dict["iterations"] = iterations;
  summary_dict["brief_report"] = summary.BriefReport();
  summary_dict["full_report"] = summary.FullReport();
  return summary_dict;
}

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
}
#endif  // GAME_GAME_CERES_PYTHON_UTILS_H_
