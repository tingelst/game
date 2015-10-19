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
  QuaternionEstimation(const QuaternionEstimation& quaternion_estimation) {}
  QuaternionEstimation() {}

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

    options_.max_num_iterations = 10;
    options_.linear_solver_type = ceres::DENSE_QR;
    //    options_.function_tolerance = 10e-12;
    //    options_.parameter_tolerance = 10e-12;
    options_.num_threads = 12;
    options_.num_linear_solver_threads = 12;

    Solve(options_, &problem_, &summary_);

    //    NormalizeRotor(parameters_data);

    return parameters;
  }

  auto Summary() -> bp::dict { return game::SummaryToDict(summary_); }

  Problem problem_;
  Solver::Options options_;
  Solver::Summary summary_;
};

}  // namespace game

#endif  // GAME_GAME_QUATERNION_ESTIMATION_H_
