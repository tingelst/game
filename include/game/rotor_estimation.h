#ifndef GAME_GAME_ROTOR_ESTIMATION_H_
#define GAME_GAME_ROTOR_ESTIMATION_H_

#include <iostream>
#include <boost/numpy.hpp>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <hep/ga.hpp>
#include "game/ceres_python_utils.h"

namespace np = boost::numpy;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

namespace game {

struct RotorEstimation {
  RotorEstimation(const RotorEstimation& rotor_estimation) {}
  RotorEstimation() {}

  template <typename T>
  static void NormalizeRotor(T* array) {
    auto scale =
        static_cast<T>(1.0) / sqrt(array[0] * array[0] + array[1] * array[1] +
                                   array[2] * array[2] + array[3] * array[3]);
    array[0] *= scale;
    array[1] *= scale;
    array[2] *= scale;
    array[3] *= scale;
  }

  struct CostFunctor {
    CostFunctor(const double* a, const double* b) : a_(a), b_(b) {}

    template <typename T>
    bool operator()(const T* const r /* 4 parameters */,
                    T* residual /* 1 parameter */) const {
      using Algebra = hep::algebra<T, 3, 0>;
      using Scalar = hep::multi_vector<Algebra, hep::list<0> >;
      using Rotor = hep::multi_vector<Algebra, hep::list<0, 3, 5, 6> >;
      using Point = hep::multi_vector<Algebra, hep::list<1, 2, 4> >;
      using Vector = hep::multi_vector<Algebra, hep::list<1, 2, 4> >;

      Point a{static_cast<T>(a_[0]), static_cast<T>(a_[1]),
              static_cast<T>(a_[2])};
      Point b{static_cast<T>(b_[0]), static_cast<T>(b_[1]),
              static_cast<T>(b_[2])};

      Rotor rot{r[0], r[1], r[2], r[3]};
      NormalizeRotor(&rot[0]);

      Point rar;
      rar = hep::grade<1>(rot * a * ~rot);

      Vector dist = rar - b;

      residual[0] = dist[0];
      residual[1] = dist[1];
      residual[2] = dist[2];

      return true;
    }

   private:
    const double* a_;
    const double* b_;
  };

  static ceres::CostFunction* Create(const double* a, const double* b) {
    return (new ceres::AutoDiffCostFunction<RotorEstimation::CostFunctor, 3, 4>(
        new RotorEstimation::CostFunctor(a, b)));
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
          RotorEstimation::Create(&a_data[cols_a * i], &b_data[cols_b * i]);
      problem_.AddResidualBlock(cost_function, NULL, parameters_data);
    }

    options_.max_num_iterations = 10;
    options_.linear_solver_type = ceres::DENSE_QR;
    options_.function_tolerance = 10e-12;
    options_.parameter_tolerance = 10e-12;
    options_.num_threads = 12;
    options_.num_linear_solver_threads = 12;

    Solve(options_, &problem_, &summary_);

    NormalizeRotor(parameters_data);

    return parameters;
  }

  auto Summary() -> bp::dict { return game::SummaryToDict(summary_); }

  Problem problem_;
  Solver::Options options_;
  Solver::Summary summary_;
};

}  // namespace game

#endif  // GAME_GAME_ROTOR_ESTIMATION_H_
