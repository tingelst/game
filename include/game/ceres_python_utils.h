#ifndef GAME_GAME_CERES_PYTHON_UTILS_H_
#define GAME_GAME_CERES_PYTHON_UTILS_H_

#include <boost/python.hpp>
#include <ceres/ceres.h>

namespace bp = boost::python;

namespace game {

auto SummaryToDict(const ceres::Solver::Summary& summary) -> bp::dict {
  bp::dict summary_dict;
  summary_dict["linear_solver_type_used"] = std::string(
      ceres::LinearSolverTypeToString(summary.linear_solver_type_used));
  summary_dict["num_parameter_blocks"] = summary.num_parameter_blocks;
  summary_dict["num_parameters"] = summary.num_parameters;
  summary_dict["num_residual_blocks"] = summary.num_residual_blocks;
  summary_dict["num_residuals"] = summary.num_residuals;

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
  return summary_dict;
}
}
#endif  // GAME_GAME_CERES_PYTHON_UTILS_H_
