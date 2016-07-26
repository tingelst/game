#ifndef GAME_GAME_CERES_PYTHON_UTILS_H_
#define GAME_GAME_CERES_PYTHON_UTILS_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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

auto SummaryToDict(const ceres::Solver::Summary &summary) -> py::dict {
  py::dict summary_dict;
  summary_dict[py::str("linear_solver_type_used")] =
      py::str(ceres::LinearSolverTypeToString(summary.linear_solver_type_used));

  summary_dict[py::str("initial_cost")] = py::float_(summary.initial_cost);
  summary_dict[py::str("final_cost")] = py::float_(summary.final_cost);
  summary_dict[py::str("preprocessor_time_in_seconds")] =
    py::float_(summary.preprocessor_time_in_seconds);
  summary_dict[py::str("postprocessor_time_in_seconds")] =
    py::float_(summary.postprocessor_time_in_seconds);
  summary_dict[py::str("total_time_in_seconds")] =
      py::float_(summary.total_time_in_seconds);
  summary_dict[py::str("minimizer_time_in_seconds")] =
    py::float_(summary.minimizer_time_in_seconds);
  summary_dict[py::str("linear_solver_time_in_seconds")] =
      py::float_(summary.linear_solver_time_in_seconds);
  summary_dict[py::str("residual_evaluation_time_in_seconds")] =
    py::float_(summary.residual_evaluation_time_in_seconds);
  summary_dict[py::str("jacobian_evaluation_time_in_seconds")] =
    py::float_(summary.jacobian_evaluation_time_in_seconds);
  summary_dict[py::str("num_parameter_blocks")] =
      py::int_(summary.num_parameter_blocks);
  summary_dict[py::str("num_parameters")] = py::int_(summary.num_parameters);
  summary_dict[py::str("num_residual_blocks")] =
      py::int_(summary.num_residual_blocks);
  summary_dict[py::str("num_residuals")] = py::int_(summary.num_residuals);
  summary_dict[py::str("trust_region_strategy_type")] =
      py::str(ceres::TrustRegionStrategyTypeToString(
          summary.trust_region_strategy_type));
  summary_dict[py::str("minimizer_type")] =
      py::str(ceres::MinimizerTypeToString(summary.minimizer_type));

  py::list iterations;
  auto its = summary.iterations;
  for (int i = 0; i < its.size(); ++i) {
    py::dict iteration;
    iteration[py::str("iteration")] = py::int_(its[i].iteration);
    iteration[py::str("cost")] = py::float_(its[i].cost);
    iteration[py::str("cost_change")] = py::float_(its[i].cost_change);
    iteration[py::str("gradient_max_norm")] =
        py::float_(its[i].gradient_max_norm);
    iteration[py::str("step_norm")] = py::float_(its[i].step_norm);
    iteration[py::str("relative_decrease")] =
        py::float_(its[i].relative_decrease);
    iteration[py::str("trust_region_radius")] =
        py::float_(its[i].trust_region_radius);
    iteration[py::str("eta")] = py::float_(its[i].eta);
    iteration[py::str("linear_solver_iterations")] =
        py::int_(its[i].linear_solver_iterations);
    iterations.append(iteration);
  }
  summary_dict[py::str("iterations")] = iterations;
  summary_dict[py::str("brief_report")] =
      py::str(summary.BriefReport().c_str());
  summary_dict[py::str("full_report")] = py::str(summary.FullReport().c_str());
  return summary_dict;
}

// void SetSolverOptions(py::dict solver_options,
// ceres::Solver::Options& options) {

//[> for (auto a : solver_options) <]
//[>     std::cout << a.first << " " << a.second << std::endl; <]

//[>
//ceres::StringToLinearSolverType(std::string(py::str(solver_options[py::str("linear_solver_type")])),
//<]
//[>                                 &options.linear_solver_type); <]

// options.max_num_iterations = py::int_(solver_options["max_num_iterations"]);
// options.num_threads = py::int_(solver_options["num_threads"]);
// options.num_linear_solver_threads =
// py::int_(solver_options[py::str("num_linear_solver_threads")]);

// double parameter_tolerance{solver_options["parameter_tolerance"]};
// std::cout << parameter_tolerance << std::endl;
// options.parameter_tolerance = parameter_tolerance;

//[> options.function_tolerance =
//py::object(solver_options["function_tolerance"]).cast<double>(); <]
//[> options.gradient_tolerance = solver_options["gradient_tolerance"]; <]

//[>
//ceres::StringToTrustRegionStrategyType(std::string(py::str(solver_options["trust_region_strategy_type"])),
//<]
//[> &options.trust_region_strategy_type); <]

//[> options.minimizer_progress_to_stdout =
//solver_options["minimizer_progress_to_stdout"]; <]

//[> ceres::StringToMinimizerType( <]
//[>     std::string(py::str(solver_options["minimizer_type"])),
//&options.minimizer_type); <]

//[> std::cout << options.num_linear_solver_threads << std::endl; <]

//[>
// bp::extract<bp::list> trust_region_minimizer_iterations_to_dump(
// solver_options["trust_region_minimizer_iterations_to_dump"]);
// if (trust_region_minimizer_iterations_to_dump.check()) {
// std::vector<int> iterations_to_dump{};
// bp::list list = trust_region_minimizer_iterations_to_dump();
// for (int i = 0; i < bp::len(list); ++i) {
// iterations_to_dump.push_back(bp::extract<int>(list[i])());
//}
// options.trust_region_minimizer_iterations_to_dump = iterations_to_dump;
//}

// bp::extract<std::string> trust_region_problem_dump_directory(
// solver_options["trust_region_problem_dump_directory"]);
// if (trust_region_problem_dump_directory.check()) {
// options.trust_region_problem_dump_directory =
// trust_region_problem_dump_directory();
//}

//}
}
#endif // GAME_GAME_CERES_PYTHON_UTILS_H_
