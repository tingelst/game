#ifndef GAME_GAME_KINEMATIC_CALIBRATION_H_
#define GAME_GAME_KINEMATIC_CALIBRATION_H_

#include "game/ceres_python_utils.h"
#include "game/types.h"

namespace game {
namespace kinematic_calibration {

using ceres::AutoDiffCostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

const double kPi = 3.141592653589793238462643383279;

class KinematicCalibration {
 public:
  KinematicCalibration();
  KinematicCalibration(const KinematicCalibration& kinematic_calibration);
  explicit KinematicCalibration(const bp::dict& solver_options_dict);
  auto Run(np::ndarray qs, np::ndarray m0, np::ndarray m1, np::ndarray m2,
           np::ndarray m3, np::ndarray m4, np::ndarray m5, np::ndarray m6)
      -> void;
  auto Summary() -> bp::dict;
  static ceres::CostFunction* Create(const double* q);
  struct CostFunctor {
    CostFunctor(const double* q) : q_(q) {}

    template <typename T>
    bool operator()(const T* const m0, const T* const m1, const T* const m2,
                    const T* const m3, const T* const m4, const T* const m5,
                    const T* const m6, T* residual) const;

   private:
    const double* q_;
  };

 private:
  Problem problem_;
  Solver::Options options_;
  Solver::Summary summary_;
};

KinematicCalibration::KinematicCalibration() {}

KinematicCalibration::KinematicCalibration(
    const KinematicCalibration& kinematic_calibration) {}

KinematicCalibration::KinematicCalibration(
    const bp::dict& solver_options_dict) {
  SetSolverOptions(solver_options_dict, options_);
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

struct MotorPlus {
  template <typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
    x_plus_delta[0] = x[0] + delta[0];
    x_plus_delta[1] = x[1] + delta[1];
    x_plus_delta[2] = x[2] + delta[2];
    x_plus_delta[3] = x[3] + delta[3];
    x_plus_delta[4] = x[4] + delta[4];
    x_plus_delta[5] = x[5] + delta[5];
    x_plus_delta[6] = x[6] + delta[6];
    x_plus_delta[7] = x[7] + delta[7];

    NormalizeRotor(x_plus_delta);

    return true;
  }
};

auto KinematicCalibration::Run(np::ndarray qs, np::ndarray m0, np::ndarray m1,
                               np::ndarray m2, np::ndarray m3, np::ndarray m4,
                               np::ndarray m5, np::ndarray m6) -> void {
  CheckContiguousArray(qs, "qs");
  CheckCols(qs, "qs", 6);
  double* qs_data = reinterpret_cast<double*>(qs.get_data());

  CheckContiguousArray(m0, "m0");
  CheckContiguousArray(m1, "m1");
  CheckContiguousArray(m2, "m2");
  CheckContiguousArray(m3, "m3");
  CheckContiguousArray(m4, "m4");
  CheckContiguousArray(m5, "m5");
  CheckContiguousArray(m6, "m6");

  CheckArrayShape(m0, "m0", 8, 1);
  CheckArrayShape(m1, "m1", 8, 1);
  CheckArrayShape(m2, "m2", 8, 1);
  CheckArrayShape(m3, "m3", 8, 1);
  CheckArrayShape(m4, "m4", 8, 1);
  CheckArrayShape(m5, "m5", 8, 1);
  CheckArrayShape(m6, "m6", 8, 1);

  double* m0_data = reinterpret_cast<double*>(m0.get_data());
  double* m1_data = reinterpret_cast<double*>(m1.get_data());
  double* m2_data = reinterpret_cast<double*>(m2.get_data());
  double* m3_data = reinterpret_cast<double*>(m3.get_data());
  double* m4_data = reinterpret_cast<double*>(m4.get_data());
  double* m5_data = reinterpret_cast<double*>(m5.get_data());
  double* m6_data = reinterpret_cast<double*>(m6.get_data());

  for (int i = 0; i < qs.shape(0); ++i) {
    ceres::CostFunction* cost_function =
        KinematicCalibration::Create(&qs_data[6 * i]);
    problem_.AddResidualBlock(cost_function, NULL, m0_data, m1_data, m2_data,
                              m3_data, m4_data, m5_data, m6_data);
  }
  problem_.SetParameterization(
      m0_data, new ceres::AutoDiffLocalParameterization<MotorPlus, 8, 8>);
  problem_.SetParameterization(
      m1_data, new ceres::AutoDiffLocalParameterization<MotorPlus, 8, 8>);
  problem_.SetParameterization(
      m2_data, new ceres::AutoDiffLocalParameterization<MotorPlus, 8, 8>);
  problem_.SetParameterization(
      m3_data, new ceres::AutoDiffLocalParameterization<MotorPlus, 8, 8>);
  problem_.SetParameterization(
      m4_data, new ceres::AutoDiffLocalParameterization<MotorPlus, 8, 8>);
  problem_.SetParameterization(
      m5_data, new ceres::AutoDiffLocalParameterization<MotorPlus, 8, 8>);
  problem_.SetParameterization(
      m6_data, new ceres::AutoDiffLocalParameterization<MotorPlus, 8, 8>);

  Solve(options_, &problem_, &summary_);
}

auto KinematicCalibration::Summary() -> bp::dict {
  return game::SummaryToDict(summary_);
}

ceres::CostFunction* KinematicCalibration::Create(const double* q) {
  return (new ceres::AutoDiffCostFunction<KinematicCalibration::CostFunctor, 3,
                                          8, 8, 8, 8, 8, 8, 8>(
      new KinematicCalibration::CostFunctor(q)));
}

template <typename T>
versor::Motor<T> CreateMotorFromArray(const T* m) {
  return versor::Motor<T>{m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]};
}

template <typename T>
bool KinematicCalibration::CostFunctor::operator()(
    const T* const m0, const T* const m1, const T* const m2, const T* const m3,
    const T* const m4, const T* const m5, const T* const m6,
    T* residual) const {
  T t0m[3] = {T(1.0), T(0.0), T{1.0}};
  T t1m[3] = {T(1.0), T(0.0), T{1.0}};
  T t2m[3] = {T(1.0), T(0.0), T{1.0}};
  T t3m[3] = {T(1.0), T(0.0), T{1.0}};
  T t4m[3] = {T(1.0), T(0.0), T{1.0}};
  T t5m[3] = {T(1.0), T(0.0), T{1.0}};
  T t6m[3] = {T(1.0), T(0.0), T{1.0}};

  T r0m[4] = {cos(T(q_[0] / 2.0)), T(0.0), -sin(T(q_[0]) / 2.0), T(0.0)};
  T r1m[4] = {cos(T(q_[1] / 2.0)), T(0.0), -sin(T(q_[1]) / 2.0), T(0.0)};
  T r2m[4] = {cos(T(q_[2] / 2.0)), T(0.0), -sin(T(q_[2]) / 2.0), T(0.0)};
  T r3m[4] = {cos(T(q_[3] / 2.0)), T(0.0), -sin(T(q_[3]) / 2.0), T(0.0)};
  T r4m[4] = {cos(T(q_[4] / 2.0)), T(0.0), -sin(T(q_[4]) / 2.0), T(0.0)};
  T r5m[4] = {cos(T(q_[5] / 2.0)), T(0.0), -sin(T(q_[5]) / 2.0), T(0.0)};
  T r6m[4] = {cos(T(q_[6] / 2.0)), T(0.0), -sin(T(q_[6]) / 2.0), T(0.0)};

  versor::Motor<T> m0m{r0m[0],r0m[1],r0m[2],r0m[3]};
  versor::Motor<T> m1m;
  m1m[0] = T(1.0);
  versor::Motor<T> m2m;
  m2m[0] = T(1.0);
  versor::Motor<T> m3m;
  m3m[0] = T(1.0);
  versor::Motor<T> m4m;
  m4m[0] = T(1.0);
  versor::Motor<T> m5m;
  m5m[0] = T(1.0);
  versor::Motor<T> m6m;
  m6m[0] = T(1.0);

  versor::Motor<T> m0c = CreateMotorFromArray(const_cast<T*>(m0));
  versor::Motor<T> m1c = CreateMotorFromArray(const_cast<T*>(m1));
  versor::Motor<T> m2c = CreateMotorFromArray(const_cast<T*>(m2));
  versor::Motor<T> m3c = CreateMotorFromArray(const_cast<T*>(m3));
  versor::Motor<T> m4c = CreateMotorFromArray(const_cast<T*>(m4));
  versor::Motor<T> m5c = CreateMotorFromArray(const_cast<T*>(m5));
  versor::Motor<T> m6c = CreateMotorFromArray(const_cast<T*>(m6));

  // Error Motor

  T r_error[4] = {cos(T(kPi / 4.0)), T(0.0), -sin(T(kPi / 4.0)), T(0.0)};
  versor::Translator<T> trs_error{T(1.0), T(-0.0), T(-1.0), T(-0.0)};
  versor::Rotor<T> rot_error{r_error[0], r_error[1], r_error[2], r_error[3]};
  versor::Motor<T> m_error = (trs_error * rot_error);

  versor::Vector<T> a{T(1.0), T(0.0), T(0.0)};
  versor::Point<T> pm =
      a + versor::Scalar<T>{T(0.5)} * a * a * versor::Inf<T>{T(1.0)} + versor::Ori<T>{T(1.0)};
  versor::Point<T> pc =
      a + versor::Scalar<T>{T(0.5)} * a * a * versor::Inf<T>{T(1.0)} + versor::Ori<T>{T(1.0)};

  versor::Point<T> pm_error = pm.sp(m0m.sp(m_error));
  versor::Point<T> pc_error = pc.sp(m0c).sp(m1c).sp(m2c).sp(m3c).sp(m4c).sp(m5c).sp(m6c);

  residual[0] = pc_error[0] - pm_error[0];
  residual[1] = pc_error[1] - pm_error[1];
  residual[2] = pc_error[2] - pm_error[2];

  return true;
}

}  // namespace kinematic_calibration
}  // namespace game

#endif  // GAME_GAME_KINEMATIC_CALIBRATION_H_
