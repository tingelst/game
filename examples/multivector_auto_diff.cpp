// GAME - Geometric Algebra Multivector Estimation
//
// Copyright (c) 2015, Norwegian University of Science and Technology
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of GAME nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVE CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "ceres/autodiff_local_parameterization.h"
#include <Eigen/Core>
#include <ceres/autodiff_cost_function.h>
#include <game/vsr/cga_op.h>
#include <glog/logging.h>

using namespace vsr::cga;

const double kPi = 3.141592653589793238462643383279;

struct VectorCorrespondencesCostFunctor {
  VectorCorrespondencesCostFunctor(const Vec &a, const Vec &b) : a_(a), b_(b) {}

  template <typename T>
  auto operator()(const T *const rotor, T *residual) const -> bool {
    Rotor<T> R(rotor);
    Vector<T> a(a_);
    Vector<T> b(b_);
    Vector<T> c = a.spin(R);

    for (int i = 0; i < 3; ++i) {
      residual[i] = c[i] - b[i];
    }

    return true;
  }

private:
  const Vec a_;
  const Vec b_;
};

struct RotorPlus {
  template <typename T>
  bool operator()(const T *x, const T *delta, T *x_plus_delta) const {
    const T squared_norm_delta =
        delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
    T r_delta[4];
    if (squared_norm_delta > T(0.0)) {
      T norm_delta = sqrt(squared_norm_delta);
      const T sin_delta_by_delta = sin(norm_delta) / norm_delta;
      r_delta[0] = cos(norm_delta);
      r_delta[1] = sin_delta_by_delta * delta[0];
      r_delta[2] = sin_delta_by_delta * delta[1];
      r_delta[3] = sin_delta_by_delta * delta[2];
    } else {
      // We do not just use r_delta = [1,0,0,0] here because that is a
      // constant and when used for automatic differentiation will
      // lead to a zero derivative. Instead we take a first order
      // approximation and evaluate it at zero.
      r_delta[0] = T(1.0);
      r_delta[1] = delta[0];
      r_delta[2] = delta[1];
      r_delta[3] = delta[2];
    }

    Rotor<T> rotor = Rotor<T>{r_delta[0], r_delta[1], r_delta[2], r_delta[3]} *
                     Rotor<T>{x[0], x[1], x[2], x[3]};

    for (int i = 0; i < 4; ++i)
      x_plus_delta[i] = rotor[i];

    return true;
  }
};

int main(int argc, char **argv) {

  google::InitGoogleLogging(argv[0]);

  double theta_half{kPi / 6.0};
  Rot rotor{cos(theta_half), -sin(theta_half), 0.0, 0.0};
  Vec a{1.0, 0.0, 0.0};
  Vec b{0.0, 1.0, 0.0};

  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> global_jacobian;
  Eigen::Matrix<double, 4, 3, Eigen::RowMajor> local_jacobian;
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> jacobian;
  Eigen::Matrix<double, 1, 3> result;

  const double *parameters[1] = {rotor.begin()};
  double *global_jacobian_array[1] = {global_jacobian.data()};

  ceres::AutoDiffCostFunction<VectorCorrespondencesCostFunctor, 3, 4>(
      new VectorCorrespondencesCostFunctor(a, b))
      .Evaluate(parameters, result.data(), global_jacobian_array);

  ceres::AutoDiffLocalParameterization<RotorPlus, 4, 3>(new RotorPlus())
      .ComputeJacobian(rotor.begin(), local_jacobian.data());

  jacobian = global_jacobian * local_jacobian;

  std::cout << "Jacobian of the function F = R * a * ~R - b where" << std::endl;
  std::cout << "R is a Euclidean rotor with coefficients:" << std::endl;
  std::cout << "R: " << rotor << std::endl;
  std::cout << "and a and b are vectors with coefficients:" << std::endl;
  std::cout << "a: " << a << std::endl;
  std::cout << "b: " << b << std::endl;
  std::cout << "The resulting vector have coefficients:" << std::endl;
  std::cout << result << std::endl;
  std::cout << std::endl;
  std::cout << "The resulting 3x4 global jacobian:" << std::endl;
  std::cout << global_jacobian << std::endl;
  std::cout << "The 3x3 local jacobian:" << std::endl;
  std::cout << local_jacobian << std::endl;
  std::cout << "The final 3x3 jacobian:" << std::endl;
  std::cout << jacobian << std::endl;

  return 0;
}
