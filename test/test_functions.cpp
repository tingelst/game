#define CATCH_CONFIG_MAIN
#include "catch.hpp"

// Use the Adept autodiff library
#include "adept.h"

#include <cmath>
#include "ceres/autodiff_local_parameterization.h"
#include "ceres/fpclassify.h"
#include "ceres/internal/autodiff.h"
#include "ceres/internal/eigen.h"
#include "ceres/local_parameterization.h"
#include "ceres/rotation.h"

#include "game/adept_autodiff_local_parameterization.h"

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
  stack.jacobian(&ajacobian[0], true);

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
      REQUIRE(ceres::MatrixRef(ajacobian, kGlobalSize, kLocalSize)(j, i) ==
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

struct IdentityPlus {
  template <typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
    for (int i = 0; i < 3; ++i) {
      x_plus_delta[i] = x[i] + delta[i];
    }
    return true;
  }
};

TEST_CASE("IdentityParameterization", "AutoDiffLocalParameterizationTest") {
  ceres::AutoDiffLocalParameterization<IdentityPlus, 3, 3> parameterization;

  double x[3] = {1.0, 2.0, 3.0};
  double delta[3] = {0.0, 1.0, 2.0};
  double x_plus_delta[3] = {0.0, 0.0, 0.0};
  parameterization.Plus(x, delta, x_plus_delta);

  REQUIRE(x_plus_delta[0] == 1.0);
  REQUIRE(x_plus_delta[1] == 3.0);
  REQUIRE(x_plus_delta[2] == 5.0);

  double jacobian[9];
  parameterization.ComputeJacobian(x, jacobian);
  int k = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j, ++k) {
      double tmp = (i == j) ? 1.0 : 0.0;
      REQUIRE(jacobian[k] == tmp);
    }
  }

  std::cout << "Ceres IdentityParamaterization:\n";
  std::cout << ceres::MatrixRef(jacobian, parameterization.GlobalSize(),
                                parameterization.LocalSize()) << std::endl;
}

#include "game/adept_autodiff_local_parameterization.h"
TEST_CASE("AdeptIdentityParameterization",
          "AdeptAutoDiffLocalParameterizationTest") {
  game::AdeptAutoDiffLocalParameterization<IdentityPlus, 3, 3> parameterization;

  double x[3] = {1.0, 2.0, 3.0};
  double delta[3] = {0.0, 1.0, 2.0};
  double x_plus_delta[3] = {0.0, 0.0, 0.0};
  parameterization.Plus(x, delta, x_plus_delta);

  REQUIRE(x_plus_delta[0] == 1.0);
  REQUIRE(x_plus_delta[1] == 3.0);
  REQUIRE(x_plus_delta[2] == 5.0);

  double jacobian[9];
  parameterization.ComputeJacobian(x, jacobian);
  int k = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j, ++k) {
      double tmp = (i == j) ? 1.0 : 0.0;
      REQUIRE(jacobian[k] == tmp);
    }
  }

  std::cout << "Adept IdentityParameterization:\n";
  std::cout << ColMajorMatrixRef(jacobian, parameterization.GlobalSize(),
                                 parameterization.LocalSize()) << std::endl;
}

struct ScaledPlus {
  explicit ScaledPlus(const double& scale_factor)
      : scale_factor_(scale_factor) {}

  template <typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
    for (int i = 0; i < 3; ++i) {
      x_plus_delta[i] = x[i] + T(scale_factor_) * delta[i];
    }
    return true;
  }

  const double scale_factor_;
};

TEST_CASE("ScaledParameterization", "AutoDiffLocalParameterizationTest") {
  const double kTolerance = 1e-14;

  ceres::AutoDiffLocalParameterization<ScaledPlus, 3, 3> parameterization(
      new ScaledPlus(1.2345));

  double x[3] = {1.0, 2.0, 3.0};
  double delta[3] = {0.0, 1.0, 2.0};
  double x_plus_delta[3] = {0.0, 0.0, 0.0};
  parameterization.Plus(x, delta, x_plus_delta);

  REQUIRE(x_plus_delta[0] == Approx(1.0).epsilon(kTolerance));
  REQUIRE(x_plus_delta[1] == Approx(3.2345).epsilon(kTolerance));
  REQUIRE(x_plus_delta[2] == Approx(5.469).epsilon(kTolerance));

  double jacobian[9];
  parameterization.ComputeJacobian(x, jacobian);
  int k = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j, ++k) {
      double tmp = (i == j) ? 1.2345 : 0.0;
      REQUIRE(jacobian[k] == Approx(tmp).epsilon(kTolerance));
    }
  }

  std::cout << "ceres::ScaledParamaterization:\n";
  std::cout << ceres::MatrixRef(jacobian, parameterization.GlobalSize(),
                                parameterization.LocalSize()) << std::endl;
}

TEST_CASE("AdeptScaledParameterization",
          "AdeptAutoDiffLocalParameterizationTest") {
  const double kTolerance = 1e-14;

  game::AdeptAutoDiffLocalParameterization<ScaledPlus, 3, 3> parameterization(
      new ScaledPlus(1.2345));

  double x[3] = {1.0, 2.0, 3.0};
  double delta[3] = {0.0, 1.0, 2.0};
  double x_plus_delta[3] = {0.0, 0.0, 0.0};
  parameterization.Plus(x, delta, x_plus_delta);

  REQUIRE(x_plus_delta[0] == Approx(1.0).epsilon(kTolerance));
  REQUIRE(x_plus_delta[1] == Approx(3.2345).epsilon(kTolerance));
  REQUIRE(x_plus_delta[2] == Approx(5.469).epsilon(kTolerance));

  double jacobian[9];
  parameterization.ComputeJacobian(x, jacobian);
  int k = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j, ++k) {
      double tmp = (i == j) ? 1.2345 : 0.0;
      REQUIRE(jacobian[k] == Approx(tmp).epsilon(kTolerance));
    }
  }

  std::cout << "AdeptScaledParameterization:\n";
  std::cout << ColMajorMatrixRef(jacobian, parameterization.GlobalSize(),
                                 parameterization.LocalSize()) << std::endl;
}

void QuaternionParameterizationTestHelper(const double* x,
                                          const double* delta) {
  const double kTolerance = 1e-14;
  const int kGlobalSize = 4;
  const int kLocalSize = 3;
  double x_plus_delta_ref[4] = {0.0, 0.0, 0.0, 0.0};
  double jacobian_ref[12];

  ceres::QuaternionParameterization ref_parameterization;
  ref_parameterization.Plus(x, delta, x_plus_delta_ref);
  ref_parameterization.ComputeJacobian(x, jacobian_ref);

  double x_plus_delta[4] = {0.0, 0.0, 0.0, 0.0};
  double jacobian[12];
  ceres::AutoDiffLocalParameterization<QuaternionPlus, 4, 3> parameterization;
  parameterization.Plus(x, delta, x_plus_delta);
  parameterization.ComputeJacobian(x, jacobian);

  double ax_plus_delta[4] = {0.0, 0.0, 0.0, 0.0};
  double ajacobian[12];
  game::AdeptAutoDiffLocalParameterization<QuaternionPlus, 4, 3>
      aparameterization;
  aparameterization.Plus(x, delta, ax_plus_delta);
  aparameterization.ComputeJacobian(x, ajacobian);

  for (int i = 0; i < kGlobalSize; ++i) {
    REQUIRE(x_plus_delta[i] == Approx(x_plus_delta_ref[i]).epsilon(kTolerance));
    REQUIRE(ax_plus_delta[i] ==
            Approx(x_plus_delta_ref[i]).epsilon(kTolerance));
  }

  const double x_plus_delta_norm = sqrt(
      x_plus_delta[0] * x_plus_delta[0] + x_plus_delta[1] * x_plus_delta[1] +
      x_plus_delta[2] * x_plus_delta[2] + x_plus_delta[3] * x_plus_delta[3]);

  REQUIRE(x_plus_delta_norm == Approx(1.0).epsilon(kTolerance));

  for (int i = 0; i < kLocalSize; ++i) {
    for (int j = 0; j < kGlobalSize; ++j) {
      REQUIRE(ceres::IsFinite(jacobian[i]) == true);
      REQUIRE(ceres::MatrixRef(jacobian, kGlobalSize, kLocalSize)(j, i) ==
              Approx(ceres::MatrixRef(jacobian_ref, kGlobalSize, kLocalSize)(
                         j, i)).epsilon(kTolerance));
      REQUIRE(ceres::MatrixRef(ajacobian, kGlobalSize, kLocalSize)(j, i) ==
              Approx(ceres::MatrixRef(jacobian_ref, kGlobalSize, kLocalSize)(
                         j, i)).epsilon(kTolerance));
    }
  }

  std::cout << "CeresQuaternionAutodiff:\n";
  std::cout << ceres::MatrixRef(jacobian, kGlobalSize, kLocalSize) << std::endl;

  std::cout << "AdeptQuaternionAutodiff:\n";
  std::cout << ColMajorMatrixRef(ajacobian, kGlobalSize, kLocalSize)
            << std::endl;
}

TEST_CASE("QuaternionParameterizationZeroTest2",
          "AutoDiffLocalParameterization") {
  double x[4] = {0.5, 0.5, 0.5, 0.5};
  double delta[3] = {0.0, 0.0, 0.0};
  QuaternionParameterizationTestHelper(x, delta);
}

TEST_CASE("QuaternionParameterizationNearZeroTest2", "AutoDiffLocalParameterization") {
  double x[4] = {0.52, 0.25, 0.15, 0.45};
  double norm_x = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
  for (int i = 0; i < 4; ++i) {
    x[i] = x[i] / norm_x;
  }

  double delta[3] = {0.24, 0.15, 0.10};
  for (int i = 0; i < 3; ++i) {
    delta[i] = delta[i] * 1e-14;
  }

  QuaternionParameterizationTestHelper(x, delta);
}

TEST_CASE("QuaternionParameterizationNonZeroTest2", "AutoDiffLocalParameterization") {
  double x[4] = {0.52, 0.25, 0.15, 0.45};
  double norm_x = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);

  for (int i = 0; i < 4; ++i) {
    x[i] = x[i] / norm_x;
  }

  double delta[3] = {0.24, 0.15, 0.10};
  QuaternionParameterizationTestHelper(x, delta);
}
