//
// Created by lars on 17.11.15.
//

#ifndef GAME_GAME_MOTOR_PARAMETERIZATION_H_
#define GAME_GAME_MOTOR_PARAMETERIZATION_H_

#include "game/vsr/cga_types.h"

namespace game {

#define CASESTR(x) \
  case x:          \
    return #x
#define STRENUM(x)   \
  if (value == #x) { \
    *type = x;       \
    return true;     \
  }


enum MotorParameterizationType {
  NORMALIZE,
};

struct MotorPolarDecomposition {
  template <typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const {

    using vsr::cga::Scalar;
    using vsr::cga::Motor;

    T a[8];
    for (int i = 0; i < 8; ++i) {
      a[i] = x[i] + delta[i];
    }

    Motor<T> X{a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]};

    T norm = X.norm();

    Motor<T> b = X * ~X;

    T s0 = b[0];
    T s4 = b[7];

    Motor<T> M = X * Scalar<T>{(T(1.0) - (s4 / (T(2.0) * s0))) / norm};
    for (int i = 0; i < 8; ++i) {
      x_plus_delta[i] = M[i];
    }

    return true;
  }
};

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

struct MotorNormalizeRotorPlus {
  template <typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
    std::cout << "normalization parameterization" << std::endl;

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

struct MotorPlus {
  template <typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
    std::cout << "no parameterization" << std::endl;

    x_plus_delta[0] = x[0] + delta[0];
    x_plus_delta[1] = x[1] + delta[1];
    x_plus_delta[2] = x[2] + delta[2];
    x_plus_delta[3] = x[3] + delta[3];
    x_plus_delta[4] = x[4] + delta[4];
    x_plus_delta[5] = x[5] + delta[5];
    x_plus_delta[6] = x[6] + delta[6];
    x_plus_delta[7] = x[7] + delta[7];

    return true;
  }
};

}  // namespace game

#endif  // GAME_GAME_MOTOR_PARAMETERIZATION_H_
