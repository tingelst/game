#ifndef GAME_GAME_RIGID_BODY_MOTION_ESTIMATION_H_
#define GAME_GAME_RIGID_BODY_MOTION_ESTIMATION_H_

#include <iostream>
#include <boost/numpy.hpp>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <hep/ga.hpp>

#include "game/types.h"
#include "game/ceres_python_utils.h"

namespace np = boost::numpy;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

namespace game {

struct GeneralRotorPlus {
  template <typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
    // x is general rotor
    // delta is a dual line
    // x_plus_delta = exp(delta) * x
  }
};

struct GeneralRotorEstimation {
  GeneralRotorEstimation(
      const GeneralRotorEstimation& general_rotor_estimation) {}
  GeneralRotorEstimation() {}

  struct CostFunctor {
    CostFunctor(const double* a, const double* b) : a_(a), b_(b) {}

    template <typename T>
    bool operator()(const T* const g /* general_rotation: 7 parameters */,
                    T* residual /* 1 parameter */) const {
      cga::Infty<T> ni{static_cast<T>(1.0), static_cast<T>(1.0)};
      cga::Orig<T> no{static_cast<T>(-0.5), static_cast<T>(0.5)};

      // Euclidean basis
      cga::E1<T> e1{static_cast<T>(1.0)};
      cga::E2<T> e2{static_cast<T>(1.0)};
      cga::E3<T> e3{static_cast<T>(1.0)};

      cga::Scalar<T> half{static_cast<T>(0.5)};
      cga::Scalar<T> one{static_cast<T>(1.0)};

      cga::EuclideanPoint<T> E3A{static_cast<T>(a_[0]), static_cast<T>(a_[1]),
                                 static_cast<T>(a_[2])};
      cga::EuclideanPoint<T> E3B{static_cast<T>(b_[0]), static_cast<T>(b_[1]),
                                 static_cast<T>(b_[2])};
      cga::Point<T> C3A = hep::grade<1>(E3A + half * E3A * E3A * ni + no);
      cga::Point<T> C3B = hep::grade<1>(E3B + half * E3B * E3B * ni + no);

      const T scale =
          T(1.0) / sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2] + g[3] * g[3]);

      cga::GeneralRotor<T> general_rotor =
          //      cga::Motor<T> general_rotor =
          cga::Rotor<T>{scale * g[0], scale * g[1], scale * g[2],
                        scale * g[3]} +
          g[4] * e1 * ni + g[5] * e2 * ni + g[6] * e3 * ni;
      //                       + g[7] * e1 * e2 * e3 * ni;

      cga::Point<T> RARrev =
          hep::eval(hep::grade<1>(general_rotor * C3A * ~general_rotor));

      residual[0] = RARrev[0] - C3B[0];
      residual[1] = RARrev[1] - C3B[1];
      residual[2] = RARrev[2] - C3B[2];

      return true;
    }

   private:
    const double* a_;
    const double* b_;
  };

  static ceres::CostFunction* Create(const double* a, const double* b) {
    return (new ceres::AutoDiffCostFunction<GeneralRotorEstimation::CostFunctor,
                                            3, 7>(
        new GeneralRotorEstimation::CostFunctor(a, b)));
  }

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

  np::ndarray Run(np::ndarray general_rotor, np::ndarray a, np::ndarray b) {
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
    auto rows_general_rotor = general_rotor.shape(0);
    auto cols_general_rotor = general_rotor.shape(1);
    if (!((rows_general_rotor == 7) && (cols_general_rotor == 1))) {
      throw "parameter array must have shape (3,1)";
    }

    if (!((rows_a == rows_b) && (cols_a == cols_b))) {
      throw "input array a and b must have the same shape";
    }

    double* general_rotor_data =
        reinterpret_cast<double*>(general_rotor.get_data());
    double* a_data = reinterpret_cast<double*>(a.get_data());
    double* b_data = reinterpret_cast<double*>(b.get_data());

    for (int i = 0; i < rows_a; ++i) {
      ceres::CostFunction* cost_function =
          Create(&a_data[cols_a * i], &b_data[cols_b * i]);
      problem_.AddResidualBlock(cost_function, NULL, general_rotor_data);
    }

    options_.max_num_iterations = 100;
    options_.linear_solver_type = ceres::DENSE_QR;
    //    options_.function_tolerance = 10e-12;
    //    options_.parameter_tolerance = 10e-12;
    options_.num_threads = 12;
    options_.num_linear_solver_threads = 12;

    Solve(options_, &problem_, &summary_);

    NormalizeRotor(general_rotor_data);

    rotor_translation_[0] = general_rotor_data[0];
    rotor_translation_[1] = general_rotor_data[1];
    rotor_translation_[2] = general_rotor_data[2];
    rotor_translation_[3] = general_rotor_data[3];
    rotor_translation_[4] = general_rotor_data[4];
    rotor_translation_[5] = general_rotor_data[5];
    rotor_translation_[6] = general_rotor_data[6];
    //    rotor_translation_[7] = general_rotor_data[7];

    np::ndarray rotor_translation_ndarray =
        np::from_data(reinterpret_cast<void*>(&rotor_translation_[0]),
                      np::dtype::get_builtin<double>(), bp::make_tuple(7, 1),
                      bp::make_tuple(8, 1), bp::object());

    return rotor_translation_ndarray;
  }

  auto Summary() -> bp::dict { return game::SummaryToDict(summary_); }

  double rotor_translation_[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Problem problem_;
  Solver::Options options_;
  Solver::Summary summary_;
};

struct RotorTranslationVectorEstimation {
  RotorTranslationVectorEstimation(const RotorTranslationVectorEstimation&
                                       rotor_translation_vector_estimation) {}
  RotorTranslationVectorEstimation() {}

  struct CostFunctor {
    CostFunctor(const double* a, const double* b) : a_(a), b_(b) {}

    template <typename T>
    bool operator()(const T* const rotor, const T* translation,
                    T* residuals) const {
      using Algebra = hep::algebra<T, 3, 0>;
      using Scalar = hep::multi_vector<Algebra, hep::list<0> >;
      using Rotor = hep::multi_vector<Algebra, hep::list<0, 3, 5, 6> >;
      using Point = hep::multi_vector<Algebra, hep::list<1, 2, 4> >;
      using Vector = hep::multi_vector<Algebra, hep::list<1, 2, 4> >;

      Point a{static_cast<T>(a_[0]), static_cast<T>(a_[1]),
              static_cast<T>(a_[2])};
      Point b{static_cast<T>(b_[0]), static_cast<T>(b_[1]),
              static_cast<T>(b_[2])};

      Rotor rot{rotor[0], rotor[1], rotor[2], rotor[3]};
      NormalizeRotor(&rot[0]);

      Vector tr{translation[0], translation[1], translation[2]};

      Point rar = hep::grade<1>(rot * a * ~rot);

      Vector dist = rar + tr - b;

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
        RotorTranslationVectorEstimation::CostFunctor, 3, 4, 3>(
        new RotorTranslationVectorEstimation::CostFunctor(a, b)));
  }

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

  np::ndarray Run(np::ndarray rotor, np::ndarray translation, np::ndarray a,
                  np::ndarray b) {
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
    auto rows_rotor = rotor.shape(0);
    auto cols_rotor = rotor.shape(1);
    if (!((rows_rotor == 4) && (cols_rotor == 1))) {
      throw "parameter array must have shape (3,1)";
    }
    auto rows_translation = translation.shape(0);
    auto cols_translation = translation.shape(1);
    if (!((rows_translation == 3) && (cols_translation == 1))) {
      throw "parameter array must have shape (3,1)";
    }

    if (!((rows_a == rows_b) && (cols_a == cols_b))) {
      throw "input array a and b must have the same shape";
    }

    double* rotor_data = reinterpret_cast<double*>(rotor.get_data());
    double* translation_data =
        reinterpret_cast<double*>(translation.get_data());
    double* a_data = reinterpret_cast<double*>(a.get_data());
    double* b_data = reinterpret_cast<double*>(b.get_data());

    for (int i = 0; i < rows_a; ++i) {
      ceres::CostFunction* cost_function =
          Create(&a_data[cols_a * i], &b_data[cols_b * i]);
      problem_.AddResidualBlock(cost_function, NULL, rotor_data,
                                translation_data);
    }

    options_.max_num_iterations = 10;
    options_.linear_solver_type = ceres::DENSE_QR;
    options_.function_tolerance = 10e-12;
    options_.parameter_tolerance = 10e-12;
    options_.num_threads = 12;
    options_.num_linear_solver_threads = 12;

    Solve(options_, &problem_, &summary_);

    NormalizeRotor(rotor_data);

    rotor_translation_[0] = rotor_data[0];
    rotor_translation_[1] = rotor_data[1];
    rotor_translation_[2] = rotor_data[2];
    rotor_translation_[3] = rotor_data[3];
    rotor_translation_[4] = translation_data[0];
    rotor_translation_[5] = translation_data[1];
    rotor_translation_[6] = translation_data[2];

    np::ndarray rotor_translation_ndarray =
        np::from_data(reinterpret_cast<void*>(&rotor_translation_[0]),
                      np::dtype::get_builtin<double>(), bp::make_tuple(7, 1),
                      bp::make_tuple(8, 1), bp::object());

    return rotor_translation_ndarray;
  }

  auto Summary() -> bp::dict { return game::SummaryToDict(summary_); }

  double rotor_translation_[7] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Problem problem_;
  Solver::Options options_;
  Solver::Summary summary_;
};

struct RotorTranslationVectorBivectorGeneratorEstimation {
  RotorTranslationVectorBivectorGeneratorEstimation(
      const RotorTranslationVectorBivectorGeneratorEstimation&
          rotor_translation_vector_bivector_generator_estimation) {}
  RotorTranslationVectorBivectorGeneratorEstimation() {}

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
    bool operator()(const T* const bivector, const T* translation,
                    T* residuals) const {
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
      Vector tr{translation[0], translation[1], translation[2]};

      Point rar = hep::grade<1>(rot * a * ~rot);

      Vector dist = rar + tr - b;

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
        RotorTranslationVectorBivectorGeneratorEstimation::CostFunctor, 3, 3,
        3>(new RotorTranslationVectorBivectorGeneratorEstimation::CostFunctor(
        a, b)));
  }

  np::ndarray Run(np::ndarray bivector, np::ndarray translation, np::ndarray a,
                  np::ndarray b) {
    if (!(a.get_flags() & np::ndarray::C_CONTIGUOUS)) {
      throw "input array a must be co ntiguous";
    }

    if (!(b.get_flags() & np::ndarray::C_CONTIGUOUS)) {
      throw "input array b must be contiguous";
    }

    auto rows_a = a.shape(0);
    auto cols_a = a.shape(1);
    auto rows_b = b.shape(0);
    auto cols_b = b.shape(1);
    auto rows_bivector = bivector.shape(0);
    auto cols_bivector = bivector.shape(1);
    if (!((rows_bivector == 3) && (cols_bivector == 1))) {
      throw "parameter array must have shape (3,1)";
    }
    auto rows_translation = translation.shape(0);
    auto cols_translation = translation.shape(1);
    if (!((rows_translation == 3) && (cols_translation == 1))) {
      throw "parameter array must have shape (3,1)";
    }

    if (!((rows_a == rows_b) && (cols_a == cols_b))) {
      throw "input array a and b must have the same shape";
    }

    double* bivector_data = reinterpret_cast<double*>(bivector.get_data());
    double* translation_data =
        reinterpret_cast<double*>(translation.get_data());
    double* a_data = reinterpret_cast<double*>(a.get_data());
    double* b_data = reinterpret_cast<double*>(b.get_data());

    for (int i = 0; i < rows_a; ++i) {
      ceres::CostFunction* cost_function =
          Create(&a_data[cols_a * i], &b_data[cols_b * i]);
      problem_.AddResidualBlock(cost_function, NULL, bivector_data,
                                translation_data);
    }

    options_.max_num_iterations = 10;
    options_.linear_solver_type = ceres::DENSE_QR;
    options_.function_tolerance = 10e-12;
    options_.parameter_tolerance = 10e-12;
    options_.num_threads = 12;
    options_.num_linear_solver_threads = 12;

    Solve(options_, &problem_, &summary_);

    Exp(bivector_data, &rotor_translation_[0]);

    rotor_translation_[4] = translation_data[0];
    rotor_translation_[5] = translation_data[1];
    rotor_translation_[6] = translation_data[2];

    np::ndarray rotor_translation_ndarray =
        np::from_data(reinterpret_cast<void*>(&rotor_translation_[0]),
                      np::dtype::get_builtin<double>(), bp::make_tuple(7, 1),
                      bp::make_tuple(8, 1), bp::object());

    return rotor_translation_ndarray;
  }

  auto Summary() -> bp::dict { return game::SummaryToDict(summary_); }

  double rotor_translation_[7] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Problem problem_;
  Solver::Options options_;
  Solver::Summary summary_;
};

}  // namespace game

#endif  // GAME_GAME_RIGID_BODY_MOTION_ESTIMATION_H_
