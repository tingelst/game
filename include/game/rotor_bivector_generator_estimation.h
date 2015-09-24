#ifndef GAME_GAME_ROTOR_BIVECTOR_GENERATOR_ESTIMATION_H_
#define GAME_GAME_ROTOR_BIVECTOR_GENERATOR_ESTIMATION_H_

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

struct RotorBivectorGeneratorEstimation {
  RotorBivectorGeneratorEstimation(const RotorBivectorGeneratorEstimation&
                                       rotor_bivector_generator_estimation) {}
  RotorBivectorGeneratorEstimation() {}

  template <typename T>
  static void Exp(const T* bivector, T* rotor) {
    T psi = sqrt(bivector[0] * bivector[0] + bivector[1] * bivector[1] +
                 bivector[2] * bivector[2]);
    rotor[0] = cos(psi);
    rotor[1] = -sin(psi) * (bivector[0] / psi);
    rotor[2] = -sin(psi) * (bivector[1] / psi);
    rotor[3] = -sin(psi) * (bivector[2] / psi);
  }

  struct CostFunctor {
    CostFunctor(const double* a, const double* b) : a_(a), b_(b) {}

    template <typename T>
    bool operator()(const T* const bivector, T* residuals) const {
      using Algebra = hep::algebra<T, 3, 0>;
      using Scalar = hep::multi_vector<Algebra, hep::list<0> >;
      using Rotor = hep::multi_vector<Algebra, hep::list<0, 3, 5, 6> >;
      using Point = hep::multi_vector<Algebra, hep::list<1, 2, 4> >;
      using Vector = hep::multi_vector<Algebra, hep::list<1, 2, 4> >;

      Point a{static_cast<T>(a_[0]), static_cast<T>(a_[1]),
              static_cast<T>(a_[2])};
      Point b{static_cast<T>(b_[0]), static_cast<T>(b_[1]),
              static_cast<T>(b_[2])};

      Rotor rot;

      Exp(bivector, &rot[0]);

      Point rar = hep::grade<1>(rot * a * ~rot);

      Vector dist = rar - b;

      residuals[0] = dist[0];
      residuals[1] = dist[1];
      residuals[2] = dist[2];

      return true;
    }

   private:
    const double* a_;
    const double* b_;
  };

  static ceres::CostFunction* Create(const double* a, const double* b) {
    return (new ceres::AutoDiffCostFunction<
        RotorBivectorGeneratorEstimation::CostFunctor, 3, 3>(
        new RotorBivectorGeneratorEstimation::CostFunctor(a, b)));
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

    if (!((rows_parameters == 3) && (cols_parameters == 1))) {
      throw "parameter array must have shape (3,1)";
    }

    if (!((rows_a == rows_b) && (cols_a == cols_b))) {
      throw "input array a and b must have the same shape";
    }

    double* parameters_data = reinterpret_cast<double*>(parameters.get_data());
    double* a_data = reinterpret_cast<double*>(a.get_data());
    double* b_data = reinterpret_cast<double*>(b.get_data());

    std::cout << parameters_data[0] << " ";
    std::cout << parameters_data[1] << " ";
    std::cout << parameters_data[2] << std::endl;

    for (int i = 0; i < rows_a; ++i) {
      ceres::CostFunction* cost_function =
          Create(&a_data[cols_a * i], &b_data[cols_b * i]);
      problem_.AddResidualBlock(cost_function, NULL, parameters_data);
    }

    options_.max_num_iterations = 10;
    options_.linear_solver_type = ceres::DENSE_QR;
    options_.function_tolerance = 10e-12;
    options_.parameter_tolerance = 10e-12;
    options_.num_threads = 12;
    options_.num_linear_solver_threads = 12;

    Solve(options_, &problem_, &summary_);

    Exp(parameters_data, &rotor_[0]);

    np::ndarray rotor_ndarray = np::from_data(
        reinterpret_cast<void*>(&rotor_[0]), np::dtype::get_builtin<double>(),
        bp::make_tuple(4, 1), bp::make_tuple(8, 1), bp::object());

    return rotor_ndarray;
  }

  auto Summary() -> bp::dict { return game::SummaryToDict(summary_); }

  double rotor_[4] = {1.0, 0.0, 0.0, 0.0};
  Problem problem_;
  Solver::Options options_;
  Solver::Summary summary_;
};

}  // namespace game

#endif  // GAME_GAME_ROTOR_BIVECTOR_GENERATOR_ESTIMATION_H_
