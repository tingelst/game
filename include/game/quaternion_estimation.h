#ifndef GAME_GAME_QUATERNION_ESTIMATION_H_
#define GAME_GAME_QUATERNION_ESTIMATION_H_

#include <iostream>
#include <boost/numpy.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/local_parameterization.h>
#include <glog/logging.h>

#include "game/ceres_python_utils.h"

namespace np = boost::numpy;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

namespace game {

struct QuaternionEstimation {
  QuaternionEstimation(bp::dict options) {
    SetSolverOptions(options);
  }

  QuaternionEstimation(const QuaternionEstimation& quaternion_estimation) {}
  QuaternionEstimation() {}


  void SetSolverOptions(const bp::dict &solver_options) {
    bp::extract<std::string> linear_solver_type(
        solver_options["linear_solver_type"]);
    if (linear_solver_type.check()) {
      ceres::StringToLinearSolverType(linear_solver_type(),
                                      &options_.linear_solver_type);
    }

    bp::extract<int> max_num_iterations(solver_options["max_num_iterations"]);
    if (max_num_iterations.check()) {
      options_.max_num_iterations = max_num_iterations();
    }

    bp::extract<int> num_threads(solver_options["num_threads"]);
    if (num_threads.check()) {
      options_.num_threads = num_threads();
    }

    bp::extract<int> num_linear_solver_threads(
        solver_options["num_linear_solver_threads"]);
    if (num_linear_solver_threads.check()) {
      options_.num_linear_solver_threads = num_linear_solver_threads();
    }

    bp::extract<double> parameter_tolerance(
        solver_options["parameter_tolerance"]);
    if (parameter_tolerance.check()) {
      options_.parameter_tolerance = parameter_tolerance();
    }

    bp::extract<double> function_tolerance(
        solver_options["function_tolerance"]);
    if (function_tolerance.check()) {
      options_.function_tolerance = function_tolerance();
    }

    bp::extract<std::string> trust_region_strategy_type(
        solver_options["trust_region_strategy_type"]);
    if (trust_region_strategy_type.check()) {
      ceres::StringToTrustRegionStrategyType(
          trust_region_strategy_type(), &options_.trust_region_strategy_type);
    }

    bp::extract<bool> minimizer_progress_to_stdout(
        solver_options["minimizer_progress_to_stdout"]);
    if (minimizer_progress_to_stdout.check()) {
      options_.minimizer_progress_to_stdout = minimizer_progress_to_stdout();
    }

    bp::extract<std::string> minimizer_type(solver_options["minimizer_type"]);
    if (minimizer_type.check()) {
      ceres::StringToMinimizerType(minimizer_type(), &options_.minimizer_type);
    }

    bp::extract<bp::list> trust_region_minimizer_iterations_to_dump(
        solver_options["trust_region_minimizer_iterations_to_dump"]);
    if (trust_region_minimizer_iterations_to_dump.check()) {
      std::vector<int> iterations_to_dump{};
      bp::list list = trust_region_minimizer_iterations_to_dump();
      for (int i = 0; i < bp::len(list); ++i) {
        iterations_to_dump.push_back(bp::extract<int>(list[i])());
      }
      options_.trust_region_minimizer_iterations_to_dump = iterations_to_dump;
    }

    bp::extract<std::string> trust_region_problem_dump_directory(
        solver_options["trust_region_problem_dump_directory"]);
    if (trust_region_problem_dump_directory.check()) {
      options_.trust_region_problem_dump_directory =
          trust_region_problem_dump_directory();
    }
  }

  struct CostFunctor {
    CostFunctor(const double* a, const double* b) : a_(a), b_(b) {}

    template <typename T>
    bool operator()(const T* const q /* 4 parameters */,
                    T* residual /* 1 parameter */) const {

      T a[3] = {static_cast<T>(a_[0]), static_cast<T>(a_[1]),static_cast<T>(a_[2])};
      T b[3] = {static_cast<T>(b_[0]), static_cast<T>(b_[1]),static_cast<T>(b_[2])};
      T a_rotated[3];
      ceres::UnitQuaternionRotatePoint(q, a, a_rotated);

      residual[0] = b[0] - a_rotated[0];
      residual[1] = b[1] - a_rotated[1];
      residual[2] = b[2] - a_rotated[2];

      return true;
    }

   private:
    const double* a_;
    const double* b_;
  };

  static ceres::CostFunction* Create(const double* a, const double* b) {
    return (new ceres::AutoDiffCostFunction<QuaternionEstimation::CostFunctor, 3, 4>(
        new QuaternionEstimation::CostFunctor(a, b)));
  }

  np::ndarray Run(np::ndarray parameters, np::ndarray a, np::ndarray b) {
    if (!(a.get_flags() & np::ndarray::C_CONTIGUOUS)) {
      throw "input array a must be contiguous";
    }

    if (!(b.get_flags() & np::ndarray::C_CONTIGUOUS)) {
      throw "input array b must be contiguous";
    }

    auto rows_a = a.shape(0);
    auto cols_a = a.shape(1);
    auto rows_b = b.shape(0);
    auto cols_b = b.shape(1);
    auto rows_parameters = parameters.shape(0);
    auto cols_parameters = parameters.shape(1);

    if (!((rows_parameters == 4) && (cols_parameters == 1))) {
      throw "parameter array must have shape (4,1)";
    }

    if (!((rows_a == rows_b) && (cols_a == cols_b))) {
      throw "input array a and b must have the same shape";
    }

    double* parameters_data = reinterpret_cast<double*>(parameters.get_data());
    double* a_data = reinterpret_cast<double*>(a.get_data());
    double* b_data = reinterpret_cast<double*>(b.get_data());

    for (int i = 0; i < rows_a; ++i) {
      ceres::CostFunction* cost_function =
          QuaternionEstimation::Create(&a_data[cols_a * i], &b_data[cols_b * i]);
      problem_.AddResidualBlock(cost_function, NULL, parameters_data);
    }

    ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization;
    problem_.SetParameterization(parameters_data, local_parameterization);


    Solve(options_, &problem_, &summary_);


    return parameters;
  }

  auto Summary() -> bp::dict { return game::SummaryToDict(summary_); }

  Problem problem_;
  Solver::Options options_;
  Solver::Summary summary_;
};

}  // namespace game

#endif  // GAME_GAME_QUATERNION_ESTIMATION_H_
