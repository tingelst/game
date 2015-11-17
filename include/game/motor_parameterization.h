//
// Created by lars on 17.11.15.
//

#ifndef GAME_GAME_MOTOR_PARAMETERIZATION_H_
#define GAME_GAME_MOTOR_PARAMETERIZATION_H_

namespace game {

#define CASESTR(x) case x: return #x
#define STRENUM(x) if (value == #x) { *type = x; return true;}

enum MotorParameterizationType {
  NORMALIZE,
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

} // namespace game

#endif  // GAME_GAME_MOTOR_PARAMETERIZATION_H_
