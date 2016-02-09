#include <iostream>
#include <adept.h>
#include "game/vsr/vsr.h"
#include <hep/ga.hpp>
#include <Eigen/Core>


using vsr::cga::Vector;
using vsr::cga::Point;
using vsr::cga::Rotor;
using vsr::cga::Translator;

const double kPi = 3.141592653589793238462643383279;


template<typename T>
using Matrix4 = Eigen::Matrix<T, 4, 4>;

template<typename T>
inline static Matrix4<T> s()
{
  Matrix4<T> m;
  m << T(1), T(0), T(0), T(0), T(0), T(1), T(0), T(0), T(0), T(0), T(1), T(0), T(0), T(0), T(0), T(1);
  return m;
}

template<typename T>
inline static Matrix4<T> e1()
{
  Matrix4<T> m;
  m << T(0), T(0), T(0), T(1), T(0), T(0), T(1), T(0), T(0), T(1), T(0), T(0), T(1), T(0), T(0), T(0);
  return m;
}

template<typename T>
inline static Matrix4<T> e2()
{
  Matrix4<T> m;
  m << T(0), T(0), T(1), T(0), T(0), T(0), T(0), T(-1), T(1), T(0), T(0), T(0), T(0), T(-1), T(0), T(0);
  return m;
}

template<typename T>
inline static Matrix4<T> e3()
{
  Matrix4<T> m;
  m << T(1), T(0), T(0), T(0), T(0), T(1), T(0), T(0), T(0), T(0), T(-1), T(0), T(0), T(0), T(0), T(-1);
  return m;
}

template<typename T>
inline static Matrix4<T> e23()
{
  return e2<T>() * e3<T>();
}

template<typename T>
inline static Matrix4<T> e31()
{
  return e3<T>() * e1<T>();
}

template<typename T>
inline static Matrix4<T> e12()
{
  return e1<T>() * e2<T>();
}

template<typename T>
inline static Matrix4<T> e123()
{
  return e1<T>() * e2<T>() * e3<T>();
}



template <typename T>
void Diff4(const T th, const T *a, T *b) {
  Matrix4<T> rotor = cos(T(0.5) * th) * s<T>() - sin(T(0.5) * th) * e1<T>() * e2<T>();
  Matrix4<T> rotor_inv = cos(T(0.5) * th) * s<T>() + sin(T(0.5) * th) * e1<T>() * e2<T>();
  Matrix4<T> vec_a = a[0] * e1<T>() + a[1] * e2<T>() + a[2] * e3<T>();
  Matrix4<T> vec_b = rotor * vec_a * rotor_inv;
  b[0] = vec_b(0, 3);
  b[1] = vec_b(2, 0);
  b[2] = vec_b(0, 0);
}

template <typename T>
void Diff5(const T th, const T *a, T *b) {
  using Algebra = hep::algebra<T, 3, 0>;
  using Rotor = hep::multi_vector<Algebra, hep::list<0, 3, 5, 6> >;
  using Vector = hep::multi_vector<Algebra, hep::list<1, 2, 4> >;
  Rotor rotor{cos(T(0.5) * th), -sin(T(0.5) * th), T(0.0), T(0.0)};
  Vector pnt_a{a[0], a[1], a[2]};
  Vector pnt_b = hep::grade<1>(rotor * pnt_a * ~rotor);
  for (int i = 0; i < 3; ++i) b[i] = pnt_b[i];
}




template <typename T>
void Diff(const T th, const T *a, T *b) {
  Rotor<T> rotor{cos(T(0.5) * th), -sin(T(0.5) * th), T(0.0), T(0.0)};
  Vector<T> vec_a{a[0], a[1], a[2]};
  Vector<T> vec_b = vec_a.spin(rotor);
  for (int i = 0; i < 3; ++i) b[i] = vec_b[i];
}

template <typename T>
void Diff2(const T th, const T *a, T *b) {
  Rotor<T> rotor{cos(T(0.5) * th), -sin(T(0.5) * th), T(0.0), T(0.0)};
  Point<T> pnt_a = Vector<T>{a[0], a[1], a[2]}.null();
  Point<T> pnt_b = pnt_a.spin(rotor);
  for (int i = 0; i < 5; ++i) b[i] = pnt_b[i];
}

template <typename T>
void Diff3(const T t, const T *a, T *b) {
  Translator<T> trs{T(1.0), -T(0.5) * t, T(0.0), T(0.0)};
  Point<T> pnt_a = Vector<T>{a[0], a[1], a[2]}.null();
  Point<T> pnt_b = pnt_a.spin(trs);
  for (int i = 0; i < 5; ++i) b[i] = pnt_b[i];
}

int main() {
  adept::Stack stack;
  const double theta = kPi / 3;

  double jac[5];
  adept::adouble at = 3.0;
  adept::adouble atheta = theta;
  adept::adouble a[3] = {1.0, 0.0, 0.0};  // e1
  stack.new_recording();
  adept::adouble b[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

  //Diff(atheta, a, b);
  Diff4(atheta, a, b);
  //Diff5(atheta, a, b);
  // Diff2(at, a, b);
   //Diff3(at, a, b);

  //stack.independent(&at, 1);
  stack.independent(&atheta, 1);
  stack.dependent(b, 3);
  stack.jacobian_reverse(jac);
  //stack.jacobian_forward(jac);

  for (int i = 0; i < 3; ++i) std::cout << b[i] << " " << std::endl;

  for (int i = 0; i < 3; ++i) std::cout << jac[i] << " " << std::endl;

  stack.print_status();
  //stack.print_statements();
  // stack.print_gradients();

  return 0;
}
