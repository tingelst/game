//
// Created by lars on 17.11.15.
//

#ifndef GAME_GAME_MOTOR_PARAMETERIZATION_H_
#define GAME_GAME_MOTOR_PARAMETERIZATION_H_

#include "game/vsr/cga_types.h"
#include "game/vsr/generic_op.h"

namespace game {

#define CASESTR(x)                                                             \
  case x:                                                                      \
    return #x
#define STRENUM(x)                                                             \
  if (value == #x) {                                                           \
    *type = x;                                                                 \
    return true;                                                               \
  }

enum MotorParameterizationType {
  NORMALIZE,
};

template <typename T>
void ProjectVectorOntoBivector(const T *a, const T *b, T *out) {
  T den = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
  out[0] = (b[0] * b[0] * a[0] - b[0] * b[2] * a[2] + b[1] * b[1] * a[0] +
            b[1] * b[2] * a[1]) /
           den;
  out[1] = (b[0] * b[0] * a[1] + b[0] * b[1] * a[2] + b[1] * b[2] * a[0] +
            b[2] * b[2] * a[1]) /
           den;
  out[2] = (b[0] * b[1] * a[1] - b[0] * b[2] * a[0] + b[1] * b[1] * a[2] +
            b[2] * b[2] * a[2]) /
           den;
}

template <typename T>
void RejectVectorFromBivector(const T *a, const T *b, T *out) {
  T den = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
  out[0] = b[2] * (b[0] * a[2] - b[1] * a[1] + b[2] * a[0]) / den;
  out[1] = b[1] * (-b[0] * a[2] + b[1] * a[1] - b[2] * a[0]) / den;
  out[2] = b[0] * (b[0] * a[2] - b[1] * a[1] + b[2] * a[0]) / den;
}

template <typename T> static T SquaredNorm3(const T *array) {
  return array[0] * array[0] + array[1] * array[1] + array[2] * array[2];
}

template <typename T> static T Norm3(const T *array) {
  return sqrt(SquaredNorm3(array));
}

template <typename T> static void Normalize3(const T *in, T *out) {
  auto scale = static_cast<T>(1.0) / Norm3(in);
  out[0] = in[0] / scale;
  out[1] = in[1] / scale;
  out[2] = in[2] / scale;
}

/*
struct MotorFromBivectorGeneratorNotWorking {
  template <typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
    using vsr::cga::Motor;

    Motor<T> M1;
    if (SquaredNorm3(delta) > T(0.0)) {
      T theta = Norm3(delta);
      T sin_theta = sin(theta);
      T sinc_theta = sin(theta) / theta;
      T cos_theta = cos(theta);

      T B[3];
      T v[3];
      T w[3];
      Normalize3(delta, B);
      ProjectVectorOntoBivector(&delta[3], B, v);
      RejectVectorFromBivector(&delta[3], B, w);

      T t[3];
      t[0] = cos_theta * v[0] + sinc_theta * w[0];
      t[1] = cos_theta * v[1] + sinc_theta * w[1];
      t[2] = cos_theta * v[2] + sinc_theta * w[2];

      T s = B[0] * w[2] - B[1] * w[1] + B[2] * w[0];

      M1 = Motor<T>(cos_theta, sin_theta * B[0], sin_theta * B[1],
                    sin_theta * B[2], t[0], t[1], t[2], sin_theta * s);
    } else {
      M1 = Motor<T>(T(1.0), delta[0], delta[1], delta[2], delta[3], delta[4],
                    delta[5], T(0.0));
    }

    Motor<T> M0{x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]};
    Motor<T> M2 = M1 * M0;
    for (int i = 0; i < 8; ++i) {
      x_plus_delta[i] = M2[i];
    }

    return true;
  }
};
*/

/*
struct VahlenMotorFromBivectorGenerator {
template <typename T>
bool operator()(const T *x, const T *delta, T *x_plus_delta) const {
  using vsr::cga::Scalar;
  using vsr::cga::Vector;
  using vsr::cga::Bivector;
  using vsr::cga::DualLine;
  using vsr::cga::Motor;
  using vsr::nga::Op;

  Motor<T> M1;
  if (SquaredNorm3(delta) > T(0.0)) {
    T theta = Norm3(delta);

    Scalar<T> sin_theta{sin(theta)};
    Scalar<T> sinc_theta{sin(theta) / theta};
    Scalar<T> cos_theta{cos(theta)};

    vahlen::Matrix<T> B = delta[0] * vahlen::E12<T>() +
                          delta[1] * vahlen::E13<T>() +
                          delta[2] * vahlen::E23<T>();
    B = B / theta;

    vahlen::Matrix<T> t = delta[3] * vahlen::E1<T>() +
                          delta[4] * vahlen::E2<T>() +
                          delta[5] * vahlen::E3<T>();

    vahlen::Matrix<T> tv = vahlen::ProjectVectorBivector(t, B);
    vahlen::Matrix<T> tw = vahlen::RejectVectorBivector(t, B);

    vahlen::Matrix<T> tt = cos_theta * tw + sinc_theta * tv;

    vahlen::Matrix<T> ts = B * tw;

    M1 = Motor<T>(cos_theta[0], sin_theta[0] * B(1, 0),
                  sin_theta[0] * B(2, 0), sin_theta[0] * B(3, 0), tt(2, 0),
                  tt(3, 0), tt(0, 0), sin_theta[0] * ts(1, 0));

  } else {
    T m_arr[8] = {T(1.0),   delta[0], delta[1], delta[2],
                  delta[3], delta[4], delta[5], T(0.0)};
    M1 = Motor<T>(m_arr);
  }

  Motor<T> M0{x};
  Motor<T> M2 = M1 * M0;
  for (int i = 0; i < 8; ++i) {
    x_plus_delta[i] = M2[i];
  }

  return true;
}
};
*/

struct MotorFromBivectorGenerator {
  template <typename T>
  bool operator()(const T *x, const T *delta, T *x_plus_delta) const {
    using vsr::cga::Scalar;
    using vsr::cga::Vector;
    using vsr::cga::Bivector;
    using vsr::cga::DualLine;
    using vsr::cga::Motor;
    using vsr::nga::Op;

    Motor<T> M1;
    if (SquaredNorm3(delta) > T(0.0)) {
      Bivector<T> B{delta[0], delta[1], delta[2]};
      T theta = B.norm();

      Scalar<T> sin_theta{sin(theta)};
      Scalar<T> sinc_theta{sin(theta) / theta};
      Scalar<T> cos_theta{cos(theta)};

      B = B.unit();

      Vector<T> t{delta[3], delta[4], delta[5]};
      Vector<T> tv = Op::project(t, B);
      Vector<T> tw = Op::reject(t, B);

      Vector<T> tt = cos_theta * tw + sinc_theta * tv;

      auto ts = B * tw;

      M1 = Motor<T>(cos_theta[0], sin_theta[0] * B[0], sin_theta[0] * B[1],
                    sin_theta[0] * B[2], tt[0], tt[1], tt[2],
                    sin_theta[0] * ts[3]);

    } else {
      M1 = Motor<T>(T(1.0), delta[0], delta[1], delta[2], delta[3], delta[4],
                    delta[5], T(0.0));
    }

    Motor<T> M0{x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]};
    Motor<T> M2 = M1 * M0;
    for (int i = 0; i < 8; ++i) {
      x_plus_delta[i] = M2[i];
    }

    return true;
  }
};

struct MotorPolarDecomposition {
  template <typename T>
  bool operator()(const T *x, const T *delta, T *x_plus_delta) const {
    using vsr::cga::Scalar;
    using vsr::cga::Motor;
    using vsr::cga::DirectionTrivector;

    T a[8];
    for (int i = 0; i < 8; ++i) {
      a[i] = x[i] + delta[i];
    }

    Motor<T> X{a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]};

    T norm = X.norm();

    Motor<T> b = X * ~X;

    T s0 = b[0];
    T s4 = b[7];

    auto Sinv =
        Scalar<T>{T(1.0) / norm} *
        (Scalar<T>{T(1.0)} + DirectionTrivector<T>{-(s4 / (T(2.0) * s0))});
    Motor<T> M = X * Sinv;

    for (int i = 0; i < 8; ++i) {
      x_plus_delta[i] = M[i];
    }

    return true;
  }
};

struct MotorTangentSpacePolarDecomposition {
  template <typename T>
  bool operator()(const T *x, const T *delta, T *x_plus_delta) const {
    using vsr::cga::Scalar;
    using vsr::cga::Motor;
    using vsr::cga::DirectionTrivector;

    Motor<T> X{x[0],
               x[1] + delta[0],
               x[2] + delta[1],
               x[3] + delta[2],
               x[4] + delta[3],
               x[5] + delta[4],
               x[6] + delta[5],
               x[7]};

    T norm = X.norm();

    Motor<T> b = X * ~X;

    T s0 = b[0];
    T s4 = b[7];

    auto Sinv =
        Scalar<T>{T(1.0) / norm} *
        (Scalar<T>{T(1.0)} + DirectionTrivector<T>{-(s4 / (T(2.0) * s0))});
    Motor<T> M = X * Sinv;

    for (int i = 0; i < 8; ++i) {
      x_plus_delta[i] = M[i];
    }

    return true;
  }
};

template <typename T> static void NormalizeRotor(T *array) {
  auto scale =
      static_cast<T>(1.0) / sqrt(array[0] * array[0] + array[1] * array[1] +
                                 array[2] * array[2] + array[3] * array[3]);
  array[0] *= scale;
  array[1] *= scale;
  array[2] *= scale;
  array[3] *= scale;
}

struct MotorNormalizeRotor {
  template <typename T>
  bool operator()(const T *x, const T *delta, T *x_plus_delta) const {
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
  bool operator()(const T *x, const T *delta, T *x_plus_delta) const {
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

#endif // GAME_GAME_MOTOR_PARAMETERIZATION_H_
