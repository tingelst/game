#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "game/versors.h"
#include "game/types.h"

#include <cmath>

namespace game {
namespace cga {

template <typename T>
cga::Motor<T> CreateMotorFromArray(T *motor) {
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
cga::Translator<T> CreateTranslatorFromArray(T *translation) {
  return T(1.0) -
         T(0.5) *
             cga::Vector<T>{translation[0], translation[1], translation[2]} *
             cga::ni<T>();
}

template <typename T>
cga::Rotor<T> CreateRotorFromArray(T *rotor) {
  return cga::Rotor<T>{rotor[0], rotor[1], rotor[2], rotor[3]};
}
}
}

TEST_CASE("Versors") {
  using namespace game::cga;

  double t_arr[3] = {1.0, 2.0, 3.0};
  double r_arr[4] = {cos(M_PI / 6.0), -sin(M_PI / 6.0), 0.0, 0.0};
  Rotor<double> r = CreateRotorFromArray(r_arr);
  Translator<double> t = CreateTranslatorFromArray(t_arr);
  double m1[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  REQUIRE(r[0] == cos(M_PI / 6.0));

  Motor<double> m = t * r;

  REQUIRE(m[0] == cos(M_PI / 6.0));
}

#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>


using namespace game::cga;
struct MotorCostFunctor {
  const double* v_;
  MotorCostFunctor(const double* v) : v_(v){}
  template <typename T>
      bool operator()(const T* const m, T* residuals) const {

    Motor<T> m = CreateMotorFromArray()

    return true;
  }
};

TEST_CASE("motors_are_slow")
{

}

using namespace ceres;
struct Cv {
  const double *v;
  Cv(const double *v) : v(v) {}
  template <typename T>
  bool operator()(const T *const q, T *residuals) const {
    UnitQuaternionRotatePoint(q, v, residuals);
    return true;
  }
};

QuaternionParameterization quaternionParameterization;


void calcJacobian(const double *q, const double *v, double *result,
                  double *qJ) {
  const double *parameters[] = {q};
  double qGlobalJ[3 * 4];
  double *jacobians[] = {qGlobalJ};
  double qLocalParamJ[4 * 3];
  AutoDiffCostFunction<Cv, 3, 4> r(new Cv(v));
  r.Evaluate(parameters, result, jacobians);
  quaternionParameterization.ComputeJacobian(q, qLocalParamJ);

  quaternionParameterization.ComputeJacobian()

  internal::MatrixMatrixMultiply<3, 4, 4, 3, 0>(qLocalParamJ, 3, 4, localParamJ,
                                                4, 3, qJ, 0, 0, 3, 3);
}
