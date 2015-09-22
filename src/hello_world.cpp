#include <iostream>

#include <boost/numpy.hpp>
#include <boost/python.hpp>
#include <ceres/ceres.h>

namespace bp = boost::python;
namespace np = boost::numpy;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

struct OptimizationProblem {
  OptimizationProblem(const OptimizationProblem& optimization_problem){};
  OptimizationProblem(){};

  struct CostFunctor {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
      residual[0] = T(10.0) - x[0];
      return true;
    }
  };

  void Run(np::ndarray arr) {
    double* x = reinterpret_cast<double*>(arr.get_data());
    const double initial_x = x[0];
    CostFunction* cost_function =
        new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    problem_.AddResidualBlock(cost_function, NULL, x);
    options_.minimizer_progress_to_stdout = false;
    Solve(options_, &problem_, &summary_);
  }

  auto Summary() -> bp::dict {
    bp::dict summary;
    summary["linear_solver_type_used"] = std::string(
        ceres::LinearSolverTypeToString(summary_.linear_solver_type_used));
    summary["num_parameter_blocks"] = summary_.num_parameter_blocks;
    summary["num_parameters"] = summary_.num_parameters;
    summary["num_residual_blocks"] = summary_.num_residual_blocks;
    summary["num_residuals"] = summary_.num_residuals;

    bp::list iterations;
    auto its = summary_.iterations;
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
    summary["iterations"] = iterations;
    summary["brief_report"] = summary_.BriefReport();
    return summary;
  }

  Problem problem_;
  Solver::Options options_;
  Solver::Summary summary_;
};

BOOST_PYTHON_MODULE(libhello_world) {
  np::initialize();
  bp::class_<OptimizationProblem>("OptimizationProblem")
      .def("run", &OptimizationProblem::Run)
      .def("summary", &OptimizationProblem::Summary);
}
