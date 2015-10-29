#define CATCH_CONFIG_MAIN
#include "catch.hpp"

// Use the Adept autodiff library
#include "adept_source.h"

#include <cmath>
#include "ceres/autodiff_local_parameterization.h"
#include "ceres/fpclassify.h"
#include "ceres/internal/autodiff.h"
#include "ceres/internal/eigen.h"
#include "ceres/local_parameterization.h"
#include "ceres/rotation.h"

// Functor needed to implement automatically differentiated Plus for
// quaternions.
struct QuaternionPlus {
  template <typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
    const T squared_norm_delta =
        delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];

    T q_delta[4];
    if (squared_norm_delta > T(0.0)) {
      T norm_delta = sqrt(squared_norm_delta);
      const T sin_delta_by_delta = sin(norm_delta) / norm_delta;
      q_delta[0] = cos(norm_delta);
      q_delta[1] = sin_delta_by_delta * delta[0];
      q_delta[2] = sin_delta_by_delta * delta[1];
      q_delta[3] = sin_delta_by_delta * delta[2];
    } else {
      // We do not just use q_delta = [1,0,0,0] here because that is a
      // constant and when used for automatic differentiation will
      // lead to a zero derivative. Instead we take a first order
      // approximation and evaluate it at zero.
      q_delta[0] = T(1.0);
      q_delta[1] = delta[0];
      q_delta[2] = delta[1];
      q_delta[3] = delta[2];
    }

    ceres::QuaternionProduct(q_delta, x, x_plus_delta);
    return true;
  }
};

typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::ColMajor>> ColMajorMatrixRef;

void QuaternionParameterizationTestHelper(const double* x, const double* delta,
                                          const double* q_delta) {
  const int kGlobalSize = 4;
  const int kLocalSize = 3;

  const double kTolerance = 1e-14;
  double x_plus_delta_ref[kGlobalSize] = {0.0, 0.0, 0.0, 0.0};
  ceres::QuaternionProduct(q_delta, x, x_plus_delta_ref);

  double x_plus_delta[kGlobalSize] = {0.0, 0.0, 0.0, 0.0};
  ceres::QuaternionParameterization parameterization;
  parameterization.Plus(x, delta, x_plus_delta);
  for (int i = 0; i < kGlobalSize; ++i) {
    REQUIRE(x_plus_delta[i] == Approx(x_plus_delta_ref[i]).epsilon(kTolerance));
  }

  const double x_plus_delta_norm = sqrt(
      x_plus_delta[0] * x_plus_delta[0] + x_plus_delta[1] * x_plus_delta[1] +
      x_plus_delta[2] * x_plus_delta[2] + x_plus_delta[3] * x_plus_delta[3]);

  REQUIRE(x_plus_delta_norm == Approx(1.0).epsilon(kTolerance));

  double jacobian_ref[12];
  double zero_delta[kLocalSize] = {0.0, 0.0, 0.0};
  const double* parameters[2] = {x, zero_delta};
  double* jacobian_array[2] = {NULL, jacobian_ref};

  // Autodiff jacobian at delta_x = 0.
  ceres::internal::AutoDiff<QuaternionPlus, double, kGlobalSize,
                            kLocalSize>::Differentiate(QuaternionPlus(),
                                                       parameters, kGlobalSize,
                                                       x_plus_delta,
                                                       jacobian_array);
  adept::Stack stack;

  adept::adouble ax[kGlobalSize];
  adept::adouble adelta[kLocalSize];

  adept::set_values(&ax[0], kGlobalSize, x);
  adept::set_values(&adelta[0], kLocalSize, zero_delta);

  stack.new_recording();
  adept::adouble ax_plus_delta[kGlobalSize];

  QuaternionPlus adept_quaternion_plus;
  adept_quaternion_plus(ax, adelta, ax_plus_delta);

  stack.independent(&adelta[0], kLocalSize);
  stack.dependent(&ax_plus_delta[0], kGlobalSize);

  double ajacobian[kGlobalSize * kLocalSize];
  stack.jacobian(&ajacobian[0]);

  double jacobian[12];
  parameterization.ComputeJacobian(x, jacobian);
  std::cout << "Reference:\n";
  std::cout << ceres::MatrixRef(jacobian_ref, kGlobalSize, kLocalSize)
            << std::endl;
  std::cout << "Ceres:\n";

  std::cout << ceres::MatrixRef(jacobian, kGlobalSize, kLocalSize) << std::endl;

  std::cout << "Adept:\n";
  std::cout << ColMajorMatrixRef(ajacobian, kGlobalSize, kLocalSize)
            << std::endl;

  for (int i = 0; i < kLocalSize; ++i) {
    for (int j = 0; j < kGlobalSize; ++j) {
      REQUIRE(ceres::IsFinite(jacobian[i]) == true);
      REQUIRE(ceres::MatrixRef(jacobian, kGlobalSize, kLocalSize)(j, i) ==
              Approx(ceres::MatrixRef(jacobian_ref, kGlobalSize, kLocalSize)(
                         j, i)).epsilon(kTolerance));
      REQUIRE(ColMajorMatrixRef(ajacobian, kGlobalSize, kLocalSize)(j, i) ==
              Approx(ceres::MatrixRef(jacobian_ref, kGlobalSize, kLocalSize)(
                         j, i)).epsilon(kTolerance));
    }
  }

  ceres::Matrix global_matrix = ceres::Matrix::Random(10, kGlobalSize);
  ceres::Matrix local_matrix = ceres::Matrix::Zero(10, kLocalSize);
  parameterization.MultiplyByJacobian(x, 10, global_matrix.data(),
                                      local_matrix.data());
  ceres::Matrix expected_local_matrix =
      global_matrix * ceres::MatrixRef(jacobian, kGlobalSize, kLocalSize);
  REQUIRE((local_matrix - expected_local_matrix).norm() == 0.0);
}

TEST_CASE("ZeroTest", "[QuaternionParameterization]") {
  double x[4] = {0.5, 0.5, 0.5, 0.5};
  double delta[3] = {0.0, 0.0, 0.0};
  double q_delta[4] = {1.0, 0.0, 0.0, 0.0};
  QuaternionParameterizationTestHelper(x, delta, q_delta);
}

template <int N>
void Normalize(double* x) {
  ceres::VectorRef(x, N).normalize();
}

TEST_CASE("NearZeroTest", "[QuaternionParameterization]") {
  double x[4] = {0.52, 0.25, 0.15, 0.45};
  Normalize<4>(x);

  double delta[3] = {0.24, 0.15, 0.10};
  for (int i = 0; i < 3; ++i) {
    delta[i] = delta[i] * 1e-14;
  }

  double q_delta[4];
  q_delta[0] = 1.0;
  q_delta[1] = delta[0];
  q_delta[2] = delta[1];
  q_delta[3] = delta[2];

  QuaternionParameterizationTestHelper(x, delta, q_delta);
}

TEST_CASE("AwayFromZeroTest", "[QuaternionParameterization]") {
  double x[4] = {0.52, 0.25, 0.15, 0.45};
  Normalize<4>(x);

  double delta[3] = {0.24, 0.15, 0.10};
  const double delta_norm =
      sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
  double q_delta[4];
  q_delta[0] = cos(delta_norm);
  q_delta[1] = sin(delta_norm) / delta_norm * delta[0];
  q_delta[2] = sin(delta_norm) / delta_norm * delta[1];
  q_delta[3] = sin(delta_norm) / delta_norm * delta[2];

  QuaternionParameterizationTestHelper(x, delta, q_delta);
}

#include "game/types.h"

TEST_CASE("SpeedTest", "[MotorDifferentiation]") {
  using namespace game;
  adept::Stack stack;
  cga::Motor<adept::adouble> m0t(adept::adouble(1.0));
  m0t[0] = adept::adouble(cos(M_PI / 6.0));
  m0t[1] = adept::adouble(-sin(M_PI / 6.0));

  cga::Motor<adept::adouble> m1t(adept::adouble(1.0));
  cga::Motor<adept::adouble> m2t(adept::adouble(1.0));
  cga::Motor<adept::adouble> m3t(adept::adouble(1.0));
  cga::Motor<adept::adouble> m4t(adept::adouble(1.0));
  cga::Motor<adept::adouble> m5t(adept::adouble(1.0));

  //  cga::Motor<adept::adouble> mtest =
  //      hep::select<0, 3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23>(m0t * m1t * m2t
  //      * m3t * m4t * m5t);

  stack.new_recording();
  cga::Point<adept::adouble> pm{adept::adouble(1.0), adept::adouble(2.0),
                                adept::adouble(3.0), adept::adouble(-6.5),
                                adept::adouble(7.5)};

  std::cout << pm[0] << std::endl;
  std::cout << pm[1] << std::endl;
  std::cout << pm[2] << std::endl;
  std::cout << pm[3] << std::endl;
  std::cout << pm[4] << std::endl;

  pm = hep::grade<1>(m0t * m1t * m2t * m3t * m4t * m5t * pm * ~m0t * ~m1t *
                     ~m2t * ~m3t * ~m4t * ~m5t);

  std::cout << pm[0] << std::endl;
  std::cout << pm[1] << std::endl;
  std::cout << pm[2] << std::endl;
  std::cout << pm[3] << std::endl;
  std::cout << pm[4] << std::endl;

  stack.dependent(&pm[0], 5);
  stack.independent(&m0t[0], 12);
  stack.independent(&m1t[0], 12);
  stack.independent(&m2t[0], 12);
  stack.independent(&m3t[0], 12);
  stack.independent(&m4t[0], 12);
  stack.independent(&m5t[0], 12);

  double jacobian[5 * 6 * 12];
  stack.jacobian(&jacobian[0]);

  std::cout << ColMajorMatrixRef(jacobian, 5, 6 * 12) << std::endl;
}


TEST_CASE("SpeedTestDoubles", "[MotorDifferentiation]") {
  using namespace game;
  cga::Motor<double> m0t(1.0);
  m0t[0] = cos(M_PI / 6.0);
  m0t[1] = -sin(M_PI / 6.0);

  cga::Motor<double> m1t(1.0);
  cga::Motor<double> m2t(1.0);
  cga::Motor<double> m3t(1.0);
  cga::Motor<double> m4t(1.0);
  cga::Motor<double> m5t(1.0);

    cga::Motor<double> mtest =
        hep::select<0, 3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23>(m0t * m1t * m2t
        * m3t * m4t * m5t);

  cga::Point<double> pm{1.0, 2.0, 3.0, -6.5, 7.5};

  std::cout << pm[0] << std::endl;
  std::cout << pm[1] << std::endl;
  std::cout << pm[2] << std::endl;
  std::cout << pm[3] << std::endl;
  std::cout << pm[4] << std::endl;

  auto pm_out = hep::grade<1>(m0t * m1t * m2t * m3t * m4t * m5t * pm * ~m0t * ~m1t *
                     ~m2t * ~m3t * ~m4t * ~m5t);

  std::cout << hep::eval(pm_out)[0] << std::endl;
  std::cout << pm[1] << std::endl;
  std::cout << pm[2] << std::endl;
  std::cout << pm[3] << std::endl;
  std::cout << pm[4] << std::endl;


}
