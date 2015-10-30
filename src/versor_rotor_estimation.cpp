//
// Created by Lars Tingelstad on 28/07/15.
//

#include <ceres/ceres.h>
#include "glog/logging.h"

#include "vsr/vsr.h"

template <typename T>
using cga = vsr::algebra<vsr::metric<4, 1, true>, T>;

template <typename T>
using Scalar = vsr::GASca<cga<T>>;
template <typename T>
using Vector = vsr::GAVec<cga<T>>;
template <typename T>
using Bivector = vsr::GABiv<cga<T>>;
template <typename T>
using Rotor = vsr::GARot<cga<T>>;
template <typename T>
using Point = vsr::GAPnt<cga<T>>;
template <typename T>
using Ori = vsr::GAOri<cga<T>>;
template <typename T>
using Inf = vsr::GAInf<cga<T>>;

template <typename T>
using Motor = vsr::GAMot<cga<T>>;

using Scalard = Scalar<double>;
using Vectord = Vector<double>;
using Bivectord = Bivector<double>;
using Rotord = Rotor<double>;
using Motord = Motor<double>;

const int kNumPoints = 10;
const double a_data[] = {0.5377,  -1.3499, 1.8339,  3.0349,  -2.2588,
                         0.7254,  0.8622,  -0.0631, 0.3188,  0.7147,
                         -1.3077, -0.2050, -0.4336, -0.1241, 0.3426,
                         1.4897,  3.5784,  1.4090,  2.7694,  1.4172};
const double b_data[] = {-0.9053, -1.1838, 3.5332,  -0.0669, -0.4852,
                         2.2582,  0.3921,  -0.8339, 0.7351,  0.0810,
                         -0.8329, 1.1066,  -0.3326, 0.2749,  1.4928,
                         0.4667,  3.0641,  -2.4057, 2.6675,  -1.6339};

const double a2_data[] = {-0.1022, -0.8637, -0.2414, 0.0774,  0.3192,
                          -1.2141, 0.3129,  -1.1135, -0.8649, -0.0068,
                          -0.0301, 1.5326,  -0.1649, -0.7697, 0.6277,
                          0.3714,  1.0933,  -0.2256, 1.1093,  1.1174};

const double b2_data[] = {-0.7991, -0.3433, -0.0537, 0.2478,  -0.8919,
                          -0.8835, -0.8079, -0.8277, -0.4384, 0.7456,
                          1.3123,  0.7923,  -0.7490, -0.2420, 0.6355,
                          -0.3579, 0.3513,  -1.0596, 1.5223,  -0.4020};

const double a3_data[] = {1.4193,  -1.1480, 0.2916,  0.1049,  0.1978,
                          0.7223,  1.5877,  2.5855,  -0.8045, -0.6669,
                          0.6966,  0.1873,  0.8351,  -0.0825, -0.2437,
                          -1.9330, 0.2157,  -0.4390, -1.1658, -1.7947};

const double b3_data[] = {-0.2845, -1.8031, 0.2366,  -0.2001, 0.7244,
                          0.1898,  3.0330,  -0.0822, -0.9798, 0.3632,
                          0.5105,  -0.5096, 0.3461,  -0.7645, -1.7959,
                          -0.7554, -0.2723, -0.4063, -2.1372, 0.1123};

const double a4_trans_data[] = {0.8404,  -2.1384, -0.8880, -0.8396, 0.1001,
                                1.3546,  -0.5445, -1.0722, 0.3035,  0.9610,
                                -0.6003, 0.1240,  0.4900,  1.4367,  0.7394,
                                -1.9609, 1.7119,  -0.1977, -0.1941, -1.2078};

const double b4_trans_data[] = {0.7137, 0.1853, 0.8701, 2.3081, 3.2921,
                                2.5118, 0.7463, 1.9609, 2.9605, 2.2317,
                                1.7936, 2.5836, 3.5441, 2.2273, 0.6576,
                                0.4356, 2.7198, 0.4361, 0.7543, 1.5492};

struct DistanceError {
  DistanceError(const double* a, const double* b) : a_(a), b_(b) {}

  template <typename T>
  bool operator()(const T* const motor /* 8 parameters */,
                  T* residual /* 1 parameter */) const {
    const T scale = T(1.0) / sqrt(motor[0] * motor[0] + motor[1] * motor[1] +
                                  motor[2] * motor[2] + motor[3] * motor[3]);

    Motor<T> M{scale * motor[0], scale * motor[1], scale * motor[2],
               scale * motor[3], motor[4],         motor[5],
               motor[6],         motor[7]};

    Vector<T> a{T(a_[0]), T(a_[1]), T(0.0)};
    Point<T> pa =
        a + Scalar<T>{T(0.5)} * a * a * Inf<T>{T(1.0)} + Ori<T>{T(1.0)};
    Vector<T> b{T(b_[0]), T(b_[1]), T(0.0)};
    Point<T> pb =
        b + Scalar<T>{T(0.5)} * b * b * Inf<T>{T(1.0)} + Ori<T>{T(1.0)};

    Motor<T> M1{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    Motor<T> M2{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    Motor<T> M3{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    Motor<T> M4{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    Motor<T> M5{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    Motor<T> M6{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    Point<T> pc = (M1 * M2 * M3 * M4 * M5 * M6 * M * pa * ~M * ~M6 * ~M5 * ~M4 *
                   ~M3 * ~M2 * ~M1);

    std::cout << (Inf<T>{T(1.0)} * Ori<T>{T(1.0)})[0] << std::endl;

    residual[0] = pc[0] - pb[0];
    residual[1] = pc[1] - pb[1];
    residual[2] = pc[2] - pb[2];
    return true;
  }

  static ceres::CostFunction* Create(const double* a, const double* b) {
    return (new ceres::AutoDiffCostFunction<DistanceError, 3, 8>(
        new DistanceError(a, b)));
  }

 private:
  const double* a_;
  const double* b_;
};

void estimateMotor() {
  double t[] = {0.0, 0.0, 0.0};  // {e1,e2,e3}
  double R[] = {1, 0, 0, 0};     // {1,e12,e13,e23}
  double B[] = {0, 0, 0};        // {e12,e13,e23}

  double M[] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  ceres::Problem problem;
  for (int i = 0; i < kNumPoints; ++i) {
    ceres::CostFunction* cost_function =
        DistanceError::Create(&a4_trans_data[2 * i], &b4_trans_data[2 * i]);
    //    problem.AddResidualBlock(cost_function, NULL /* squared loss */, R,
    //    t);
    problem.AddResidualBlock(cost_function, NULL /* squared loss */, M);
  }

  ceres::Solver::Options options;
  //    options.function_tolerance = 10e-12;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  double scale =
      1.0 / sqrt(M[0] * M[0] + M[1] * M[1] + M[2] * M[2] + M[3] * M[3]);
  std::cout << Motord(scale * M[0], scale * M[1], scale * M[2], scale * M[3],
                      M[4], M[5], M[6], M[7]) << std::endl;
}

template <typename T>
T f(const T& x, const T& y) {
  return x * x + x * y;
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  estimateMotor();

  return 0;
}
