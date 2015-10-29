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
  return (
      new ceres::AutoDiffCostFunction<KinematicCalibration::CostFunctor, 3, 8, 8, 8, 8, 8, 8, 8>(
          new KinematicCalibration::CostFunctor(q)));
}

template <typename T>
cga::Motor<T> CreateMotorFromArray(T* motor) {
  cga::Infty<T> ni{static_cast<T>(1.0), static_cast<T>(1.0)};
  cga::E1<T> e1{static_cast<T>(1.0)};
  cga::E2<T> e2{static_cast<T>(1.0)};
  cga::E3<T> e3{static_cast<T>(1.0)};
  return hep::eval(
      hep::eval(cga::Rotor<T>{motor[0], motor[1], motor[2], motor[3]}) +
      hep::eval(cga::Scalar<T>{motor[4]} * e1 * ni) +
      hep::eval(cga::Scalar<T>{motor[5]} * e2 * ni) +
      hep::eval(cga::Scalar<T>{motor[6]} * e3 * ni) +
      hep::eval(cga::Scalar<T>{motor[7]} * e1 * e2 * e3 * ni));
}

template <typename T>
cga::Translator<T> CreateTranslatorFromArray(T* translation) {
  return hep::eval(
      cga::Scalar<T>{T(1.0)} -
      cga::Scalar<T>{T(0.5)} *
          cga::Vector<T>{translation[0], translation[1], translation[2]} *
          cga::ni<T>());
}

template <typename T>
cga::Rotor<T> CreateRotorFromArray(T* rotor) {
  return cga::Rotor<T>{rotor[0], rotor[1], rotor[2], rotor[3]};
}

int COUNT = 0;

template <typename T>
bool KinematicCalibration::CostFunctor::operator()(
    const T* const m0, const T* const m1, const T* const m2, const T* const m3,
    const T* const m4, const T* const m5, const T* const m6,
    T* residual) const {
  cga::Infty<T> ni{static_cast<T>(-1.0), static_cast<T>(1.0)};
  cga::Orig<T> no{static_cast<T>(0.5), static_cast<T>(0.5)};
  cga::E1<T> e1{static_cast<T>(1.0)};
  cga::E2<T> e2{static_cast<T>(1.0)};
  cga::E3<T> e3{static_cast<T>(1.0)};
  cga::Scalar<T> half{static_cast<T>(0.5)};
  cga::Scalar<T> one{static_cast<T>(1.0)};

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

  // Error Motor

  T r_error[4] = {cos(T(kPi / 6.0)), T(0.0), -sin(T(kPi / 6.0)), T(0.0)};
  T t_error[3] = {T(0.1), T(0.1), T{1.0}};
  cga::Motor<T> m_error = hep::eval(CreateTranslatorFromArray(t_error) *
                                    CreateRotorFromArray(r_error));

//  cga::Motor<T> m0m =
//      hep::eval(CreateTranslatorFromArray(t0m) * CreateRotorFromArray(r0m));
  cga::Motor<T> m1m =
      hep::eval(CreateTranslatorFromArray(t1m) * CreateRotorFromArray(r1m));
//  cga::Motor<T> m2m =
//      hep::eval(CreateTranslatorFromArray(t2m) * CreateRotorFromArray(r2m));
//  cga::Motor<T> m3m =
//      hep::eval(CreateTranslatorFromArray(t3m) * CreateRotorFromArray(r3m));
//  cga::Motor<T> m4m =
//      hep::eval(CreateTranslatorFromArray(t4m) * CreateRotorFromArray(r4m));
//  cga::Motor<T> m5m =
//      hep::eval(CreateTranslatorFromArray(t5m) * CreateRotorFromArray(r5m));
//  cga::Motor<T> m6m =
//      hep::eval(CreateTranslatorFromArray(t6m) * CreateRotorFromArray(r6m));
//  //
  //  // robot model
  //  T m0_tmp[8] = {m0[0], m0[1], m0[2], m0[3], m0[4], m0[5], m0[6], m0[7]};
  //  cga::Motor<T> m0c = CreateMotorFromArray(m0_tmp);
//  cga::Motor<T> m0c = CreateMotorFromArray(const_cast<T*>(m0));
  cga::Motor<T> m1c = CreateMotorFromArray(const_cast<T*>(m1));
//  cga::Motor<T> m2c = CreateMotorFromArray(const_cast<T*>(m2));
//  cga::Motor<T> m3c = CreateMotorFromArray(const_cast<T*>(m3));
//  cga::Motor<T> m4c = CreateMotorFromArray(const_cast<T*>(m4));
//  cga::Motor<T> m5c = CreateMotorFromArray(const_cast<T*>(m5));
//  cga::Motor<T> m6c = CreateMotorFromArray(const_cast<T*>(m6));

//  cga::Motor<T> m0cal = hep::eval(
//      hep::select<0, 3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23>(m0c * m0m * ~m0c));
  cga::Motor<T> m1cal = hep::eval(
      hep::select<0, 3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23>(m1c * m1m * ~m1c));
//  cga::Motor<T> m2cal = hep::eval(
//      hep::select<0, 3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23>(m2c * m2m * ~m2c));
//  cga::Motor<T> m3cal = hep::eval(
//      hep::select<0, 3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23>(m3c * m3m * ~m3c));
//  cga::Motor<T> m4cal = hep::eval(
//      hep::select<0, 3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23>(m4c * m4m * ~m4c));
//  cga::Motor<T> m5cal = hep::eval(
//      hep::select<0, 3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23>(m5c * m5m * ~m5c));
//  cga::Motor<T> m6cal = hep::eval(
//      hep::select<0, 3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23>(m6c * m6m * ~m6c));

//  cga::Motor<T> mm = hep::select<0, 3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23>(
//      m0m * m1m * m_error * m2m * ~m_error
//  );
//  * m3m * m4m * m5m * m6m);
//
//  cga::Motor<T> mc = hep::select<0, 3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23>(
//      m0cal * m1cal * m2cal
//  );

//      * m3cal * m4cal * m5cal * m6cal);

  cga::Point<T> pm{T(0.0), T(0.0), T(0.0), T(0.5), T(0.5)};  // no
  cga::Point<T> pc{T(0.0), T(0.0), T(0.0), T(0.5), T(0.5)};  // no


//  cga::Point<T> pm6 = hep::grade<1>(m6m * pm  * ~m6m);
//  cga::Point<T> pm5 = hep::grade<1>(m5m * pm6 * ~m5m);
//  cga::Point<T> pm4 = hep::grade<1>(m4m * pm5 * ~m4m);
//  cga::Point<T> pm3 = hep::grade<1>(m3m * pm4 * ~m3m);
//  cga::Point<T> pm2 = hep::grade<1>(m2m * pm3 * ~m2m);
//  cga::Point<T> pm1 = hep::grade<1>(m_error * m1m * ~m_error * pm * m_error * ~m1m * ~m_error);
//  cga::Point<T> pm0 = hep::grade<1>(m0m * pm1 * ~m0m);

//  cga::Point<T> pc6 = hep::grade<1>(m6cal * pc  * ~m6cal);
//  cga::Point<T> pc5 = hep::grade<1>(m5cal * pc6 * ~m5cal);
//  cga::Point<T> pc4 = hep::grade<1>(m4cal * pc5 * ~m4cal);
//  cga::Point<T> pc3 = hep::grade<1>(m3cal * pc4 * ~m3cal);
//  cga::Point<T> pc2 = hep::grade<1>(m2cal * pc3 * ~m2cal);
//  cga::Point<T> pc1 = hep::grade<1>(m1cal * pc * ~m1cal);
//  cga::Point<T> pc0 = hep::grade<1>(m0cal * pc1 * ~m0cal);

  cga::Motor<T> m0t;
  cga::Motor<T> m1t;
  cga::Motor<T> m2t;
  cga::Motor<T> m3t;
  cga::Motor<T> m4t;
  cga::Motor<T> m5t;

  auto mtest = m0t * m1t * m2t * m3t * m4t * m5t;

  cga::Point<T> ptest = hep::grade<1>(mtest * pm * ~mtest);

  cga::Point<T> pme;
  cga::Point<T> pce;

  residual[0] = pc[0] - ptest[0];
  residual[1] = pc[1] - ptest[1];
  residual[2] = pc[2] - ptest[2];

  return true;
}

}  // namespace kinematic_calibration
}  // namespace game

#endif  // GAME_GAME_KINEMATIC_CALIBRATION_H_
