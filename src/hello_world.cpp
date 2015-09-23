#include <iostream>

#include <boost/numpy.hpp>
#include <boost/python.hpp>
#include <ceres/ceres.h>

#include "game/ceres_python_utils.h"

namespace bp = boost::python;
namespace np = boost::numpy;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

namespace hello_world
{

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

  auto Summary() -> bp::dict { return game::SummaryToDict(summary_); }

  Problem problem_;
  Solver::Options options_;
  Solver::Summary summary_;
};

}

using namespace hello_world;

BOOST_PYTHON_MODULE(libhello_world) {
  np::initialize();
  bp::class_<OptimizationProblem>("OptimizationProblem")
      .def("run", &OptimizationProblem::Run)
      .def("summary", &OptimizationProblem::Summary);
}
