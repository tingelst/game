#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>
using Eigen::Matrix;
using Eigen::Dynamic;

#include "ceres/autodiff_cost_function.h"
#include "ceres/cost_function.h"
#include "ceres/internal/autodiff.h"
#include "ceres/internal/eigen.h"

#include <glog/logging.h>

#include <adept.h>

#include <game/vsr/cga_op.h>

#include <hep/ga.hpp>

#include <iostream>
#include <sstream>
#include <string>

using namespace vsr::cga;
namespace py = pybind11;

template <typename T>
void InnerProductVectorBivector(const T *vec, const T *biv, T *res) {
  Vector<T> vector(vec);
  Bivector<T> bivector(biv);
  Vector<T> result = vector <= bivector;
  for (int i = 0; i < 3; ++i)
    res[i] = result[i];
}

template <typename T>
void InnerProductVectorVector(const T *vec, const T *vec1, T *res) {
  Vector<T> vector(vec);
  Vector<T> vector1(vec1);
  Scalar<T> result = vector <= vector1;
  res[0] = result[0];
}

template <typename T>
void InnerProductVectorBivector2(const T *a, const T *b, T *res) {
  res[0] = -a[2] * b[1] - a[1] * b[0];
  res[1] = -a[2] * b[2] + a[0] * b[0];
  res[2] = a[1] * b[2] + a[0] * b[1];
}

struct InnerProductVectorBivectorFunctor2 {
  template <typename T>
  bool operator()(const T *vec, const T *biv, T *res) const {
    InnerProductVectorBivector2(vec, biv, res);
    return true;
  }
};

py::list AdeptDiffInnerProductVectorBivector(const Vec &vec, const Biv &biv) {
  auto result = py::array(py::buffer_info(
      nullptr,        /* Pointer to data (nullptr -> ask NumPy to allocate!) */
      sizeof(double), /* Size of one item */
      py::format_descriptor<double>::value(), /* Buffer format */
      2,                                      /* How many dimensions? */
      {3, 3},                 /* Number of elements for each dimension */
      {sizeof(double) * 3, 3} /* Strides for each dimension */
      ));

  auto buf = result.request();

  adept::Stack stack;

  double jac[9];
  std::array<adept::adouble, 3> vector{vec[0], vec[1], vec[2]};
  std::array<adept::adouble, 3> bivector{biv[0], biv[1], biv[2]};
  stack.new_recording();
  std::array<adept::adouble, 3> vec_ip_biv{0.0, 0.0, 0.0};

  InnerProductVectorBivector2(vector.begin(), bivector.begin(),
                              vec_ip_biv.begin());

  stack.independent(vector.begin(), 3);
  // stack.independent(bivector.begin(), 3);
  stack.dependent(vec_ip_biv.begin(), 3);

  // stack.jacobian(static_cast<double *>(buf.ptr), true);
  stack.jacobian_reverse(static_cast<double *>(buf.ptr), true);
  // stack.jacobian_forward(static_cast<double *>(buf.ptr), true);

  std::stringstream ss;
  stack.print_statements(ss);

  py::list list;
  list.append(result);
  list.append(py::str(ss.str().c_str()));

  return list;
}

struct InnerProductVectorBivectorFunctor {
  template <typename T>
  bool operator()(const T *vec, const T *biv, T *res) const {
    InnerProductVectorBivector(vec, biv, res);
    return true;
  }
};

struct InnerProductVectorVectorFunctor {
  template <typename T>
  bool operator()(const T *vec, const T *vec2, T *res) const {
    InnerProductVectorVector(vec, vec2, res);
    return true;
  }
};

py::list CeresDiffInnerProductVectorVector(const Vec &vec, const Vec &vec2) {
  auto py_array_jac = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {1, 3}, {sizeof(double) * 3, sizeof(double)}));

  auto py_array_result = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {1, 1}, {sizeof(double), sizeof(double)}));

  auto buf_jac = py_array_jac.request();
  auto buf_res = py_array_result.request();

  const double *parameters[2] = {vec.begin(), vec2.begin()};
  double *jacobians[2] = {static_cast<double *>(buf_jac.ptr), nullptr};

  ceres::AutoDiffCostFunction<InnerProductVectorVectorFunctor, 1, 3, 3>(
      new InnerProductVectorVectorFunctor())
      .Evaluate(parameters, static_cast<double *>(buf_res.ptr), jacobians);

  py::list list;
  list.append(py_array_result);
  list.append(py_array_jac);

  return list;
}
py::list CeresDiffInnerProductVectorBivector(const Vec &vec, const Biv &biv) {
  auto py_array_jac = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 3}, {sizeof(double) * 3, sizeof(double)}));

  auto py_array_result = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double) * 3, sizeof(double)}));

  auto buf_jac = py_array_jac.request();
  auto buf_res = py_array_result.request();

  const double *parameters[2] = {vec.begin(), biv.begin()};
  double *jacobians[2] = {static_cast<double *>(buf_jac.ptr), nullptr};

  ceres::AutoDiffCostFunction<InnerProductVectorBivectorFunctor, 3, 3, 3>(
      new InnerProductVectorBivectorFunctor())
      .Evaluate(parameters, static_cast<double *>(buf_res.ptr), jacobians);

  py::list list;
  list.append(py_array_result);
  list.append(py_array_jac);

  return list;
}

template <typename T> using Matrix4 = Eigen::Matrix<T, 4, 4>;

template <typename T> inline static Matrix4<T> s() {
  Matrix4<T> m;
  m << T(1), T(0), T(0), T(0), T(0), T(1), T(0), T(0), T(0), T(0), T(1), T(0),
      T(0), T(0), T(0), T(1);
  return m;
}

template <typename T> inline static Matrix4<T> e1() {
  Matrix4<T> m;
  m << T(0), T(0), T(0), T(1), T(0), T(0), T(1), T(0), T(0), T(1), T(0), T(0),
      T(1), T(0), T(0), T(0);
  return m;
}

template <typename T> inline static Matrix4<T> e2() {
  Matrix4<T> m;
  m << T(0), T(0), T(1), T(0), T(0), T(0), T(0), T(-1), T(1), T(0), T(0), T(0),
      T(0), T(-1), T(0), T(0);
  return m;
}

template <typename T> inline static Matrix4<T> e3() {
  Matrix4<T> m;
  m << T(1), T(0), T(0), T(0), T(0), T(1), T(0), T(0), T(0), T(0), T(-1), T(0),
      T(0), T(0), T(0), T(-1);
  return m;
}

struct DiffRotorMatrixFunctor {
  template <typename T> bool operator()(const T *th, const T *a, T *b) const {
    Matrix4<T> rotor =
        cos(T(0.5) * th[0]) * s<T>() - sin(T(0.5) * th[0]) * e1<T>() * e2<T>();
    Matrix4<T> rotor_inv =
        cos(T(0.5) * th[0]) * s<T>() + sin(T(0.5) * th[0]) * e1<T>() * e2<T>();
    Matrix4<T> vec_a = a[0] * e1<T>() + a[1] * e2<T>() + a[2] * e3<T>();
    Matrix4<T> vec_b = rotor * vec_a * rotor_inv;
    b[0] = vec_b(0, 3); // e1
    b[1] = vec_b(0, 2); // e2
    b[2] = vec_b(0, 0); // e3
    return true;
  }
};

py::list CeresDiffRotorMatrix(const double theta, const Vec &a) {
  auto py_array_jac = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto py_array_result = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto buf_jac = py_array_jac.request();
  auto buf_res = py_array_result.request();

  const double *parameters[2] = {&theta, a.begin()};
  double *jacobians[2] = {static_cast<double *>(buf_jac.ptr), nullptr};

  ceres::AutoDiffCostFunction<DiffRotorMatrixFunctor, 3, 1, 3>(
      new DiffRotorMatrixFunctor())
      .Evaluate(parameters, static_cast<double *>(buf_res.ptr), jacobians);

  py::list list;
  list.append(py_array_result);
  list.append(py_array_jac);

  return list;
}

struct DiffRotorVersorFunctor {
  template <typename T> bool operator()(const T *th, const T *a, T *b) const {
    Rotor<T> rotor{cos(T(0.5) * th[0]), -sin(T(0.5) * th[0]), T(0.0), T(0.0)};
    Vector<T> vec_a{a[0], a[1], a[2]};
    Vector<T> vec_b = vec_a.spin(rotor);
    for (int i = 0; i < 3; ++i)
      b[i] = vec_b[i];
    return true;
  }
};
py::list CeresDiffRotorVersor(const double theta, const Vec &a) {
  auto py_array_jac = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto py_array_result = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto buf_jac = py_array_jac.request();
  auto buf_res = py_array_result.request();

  const double *parameters[2] = {&theta, a.begin()};
  double *jacobians[2] = {static_cast<double *>(buf_jac.ptr), nullptr};

  ceres::AutoDiffCostFunction<DiffRotorVersorFunctor, 3, 1, 3>(
      new DiffRotorVersorFunctor())
      .Evaluate(parameters, static_cast<double *>(buf_res.ptr), jacobians);

  py::list list;
  list.append(py_array_result);
  list.append(py_array_jac);

  return list;
}

py::list AdeptDiffRotorMatrixForward(const double theta, const Vec &vec) {
  auto py_array_jac = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto py_array_result = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto buf_jac = py_array_jac.request();
  auto buf_res = py_array_result.request();

  adept::Stack stack;
  adept::adouble th{theta};
  std::array<adept::adouble, 3> a{vec[0], vec[1], vec[2]};
  stack.new_recording();
  std::array<adept::adouble, 3> b{0.0, 0.0, 0.0};
  DiffRotorMatrixFunctor()(&th, a.begin(), b.begin());
  stack.independent(&th, 1);
  stack.dependent(b.begin(), 3);

  // stack.jacobian(static_cast<double *>(buf_jac.ptr), true);
  // stack.jacobian_reverse(static_cast<double *>(buf_jac.ptr), true);
  stack.jacobian_forward(static_cast<double *>(buf_jac.ptr), true);

  std::stringstream ss;
  stack.print_statements(ss);

  auto res_ptr = static_cast<double *>(buf_res.ptr);
  for (int i = 0; i < 3; ++i)
    res_ptr[i] = b[i].value();

  py::list list;
  list.append(py_array_result);
  list.append(py_array_jac);
  list.append(py::str(ss.str().c_str()));
  std::stringstream sstatus;
  stack.print_status(sstatus);
  list.append(py::str(sstatus.str().c_str()));

  return list;
}

py::list AdeptDiffRotorMatrixReverse(const double theta, const Vec &vec) {
  auto py_array_jac = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto py_array_result = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto buf_jac = py_array_jac.request();
  auto buf_res = py_array_result.request();

  adept::Stack stack;
  adept::adouble th{theta};
  std::array<adept::adouble, 3> a{vec[0], vec[1], vec[2]};
  stack.new_recording();
  std::array<adept::adouble, 3> b{0.0, 0.0, 0.0};
  DiffRotorMatrixFunctor()(&th, a.begin(), b.begin());
  stack.independent(&th, 1);
  stack.dependent(b.begin(), 3);

  // stack.jacobian(static_cast<double *>(buf_jac.ptr), true);
  stack.jacobian_reverse(static_cast<double *>(buf_jac.ptr), true);
  // stack.jacobian_forward(static_cast<double *>(buf_jac.ptr), true);

  std::stringstream ss;
  stack.print_statements(ss);

  auto res_ptr = static_cast<double *>(buf_res.ptr);
  for (int i = 0; i < 3; ++i)
    res_ptr[i] = b[i].value();

  py::list list;
  list.append(py_array_result);
  list.append(py_array_jac);
  list.append(py::str(ss.str().c_str()));
  std::stringstream sstatus;
  stack.print_status(sstatus);
  list.append(py::str(sstatus.str().c_str()));

  return list;
}

py::list AdeptDiffRotorVersorForward(const double theta, const Vec &vec) {
  auto py_array_jac = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto py_array_result = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto buf_jac = py_array_jac.request();
  auto buf_res = py_array_result.request();

  adept::Stack stack;
  adept::adouble th{theta};
  std::array<adept::adouble, 3> a{vec[0], vec[1], vec[2]};
  stack.new_recording();
  std::array<adept::adouble, 3> b{0.0, 0.0, 0.0};
  DiffRotorVersorFunctor()(&th, a.begin(), b.begin());
  stack.independent(&th, 1);
  stack.dependent(b.begin(), 3);

  // stack.jacobian(static_cast<double *>(buf_jac.ptr), true);
  // stack.jacobian_reverse(static_cast<double *>(buf_jac.ptr), true);
  stack.jacobian_forward(static_cast<double *>(buf_jac.ptr), true);

  std::stringstream ss;
  stack.print_statements(ss);

  auto res_ptr = static_cast<double *>(buf_res.ptr);
  for (int i = 0; i < 3; ++i)
    res_ptr[i] = b[i].value();

  py::list list;
  list.append(py_array_result);
  list.append(py_array_jac);
  list.append(py::str(ss.str().c_str()));
  std::stringstream sstatus;
  stack.print_status(sstatus);
  list.append(py::str(sstatus.str().c_str()));

  return list;
}

py::list AdeptDiffRotorVersorReverse(const double theta, const Vec &vec) {
  auto py_array_jac = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto py_array_result = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto buf_jac = py_array_jac.request();
  auto buf_res = py_array_result.request();

  adept::Stack stack;
  adept::adouble th{theta};
  std::array<adept::adouble, 3> a{vec[0], vec[1], vec[2]};
  stack.new_recording();
  std::array<adept::adouble, 3> b{0.0, 0.0, 0.0};
  DiffRotorVersorFunctor()(&th, a.begin(), b.begin());
  stack.independent(&th, 1);
  stack.dependent(b.begin(), 3);

  // stack.jacobian(static_cast<double *>(buf_jac.ptr), true);
  stack.jacobian_reverse(static_cast<double *>(buf_jac.ptr), true);
  // stack.jacobian_forward(static_cast<double *>(buf_jac.ptr), true);

  std::stringstream ss;
  stack.print_statements(ss);

  auto res_ptr = static_cast<double *>(buf_res.ptr);
  for (int i = 0; i < 3; ++i)
    res_ptr[i] = b[i].value();

  py::list list;
  list.append(py_array_result);
  list.append(py_array_jac);
  list.append(py::str(ss.str().c_str()));
  std::stringstream sstatus;
  stack.print_status(sstatus);
  list.append(py::str(sstatus.str().c_str()));

  return list;
}

struct DiffRotorHepGAFunctor {
  template <typename T> bool operator()(const T *th, const T *a, T *b) const {
    using Algebra = hep::algebra<T, 3, 0>;
    using Rotor = hep::multi_vector<Algebra, hep::list<0, 3, 5, 6>>;
    using Vector = hep::multi_vector<Algebra, hep::list<1, 2, 4>>;
    Rotor rotor{cos(T(0.5) * th[0]), -sin(T(0.5) * th[0]), T(0.0), T(0.0)};
    Vector pnt_a{a[0], a[1], a[2]};
    Vector pnt_b = hep::grade<1>(rotor * pnt_a * ~rotor);
    for (int i = 0; i < 3; ++i)
      b[i] = pnt_b[i];
    return true;
  }
};

py::list AdeptDiffRotorHepGAForward(const double theta, const Vec &vec) {
  auto py_array_jac = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto py_array_result = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto buf_jac = py_array_jac.request();
  auto buf_res = py_array_result.request();

  adept::Stack stack;
  adept::adouble th{theta};
  std::array<adept::adouble, 3> a{vec[0], vec[1], vec[2]};
  stack.new_recording();
  std::array<adept::adouble, 3> b{0.0, 0.0, 0.0};
  DiffRotorHepGAFunctor()(&th, a.begin(), b.begin());
  stack.independent(&th, 1);
  stack.dependent(b.begin(), 3);

  // stack.jacobian(static_cast<double *>(buf_jac.ptr), true);
  // stack.jacobian_reverse(static_cast<double *>(buf_jac.ptr), true);
  stack.jacobian_forward(static_cast<double *>(buf_jac.ptr), true);

  std::stringstream ss;
  stack.print_statements(ss);

  auto res_ptr = static_cast<double *>(buf_res.ptr);
  for (int i = 0; i < 3; ++i)
    res_ptr[i] = b[i].value();

  py::list list;
  list.append(py_array_result);
  list.append(py_array_jac);
  list.append(py::str(ss.str().c_str()));
  std::stringstream sstatus;
  stack.print_status(sstatus);
  list.append(py::str(sstatus.str().c_str()));

  return list;
}

py::list AdeptDiffRotorHepGAReverse(const double theta, const Vec &vec) {
  auto py_array_jac = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto py_array_result = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto buf_jac = py_array_jac.request();
  auto buf_res = py_array_result.request();

  adept::Stack stack;
  adept::adouble th{theta};
  std::array<adept::adouble, 3> a{vec[0], vec[1], vec[2]};
  stack.new_recording();
  std::array<adept::adouble, 3> b{0.0, 0.0, 0.0};
  DiffRotorHepGAFunctor()(&th, a.begin(), b.begin());
  stack.independent(&th, 1);
  stack.dependent(b.begin(), 3);

  // stack.jacobian(static_cast<double *>(buf_jac.ptr), true);
  stack.jacobian_reverse(static_cast<double *>(buf_jac.ptr), true);
  // stack.jacobian_forward(static_cast<double *>(buf_jac.ptr), true);

  std::stringstream ss;
  stack.print_statements(ss);

  auto res_ptr = static_cast<double *>(buf_res.ptr);
  for (int i = 0; i < 3; ++i)
    res_ptr[i] = b[i].value();

  py::list list;
  list.append(py_array_result);
  list.append(py_array_jac);
  list.append(py::str(ss.str().c_str()));
  std::stringstream sstatus;
  stack.print_status(sstatus);
  list.append(py::str(sstatus.str().c_str()));

  return list;
}

py::list CeresDiffRotorHepGA(const double theta, const Vec &a) {
  auto py_array_jac = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto py_array_result = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto buf_jac = py_array_jac.request();
  auto buf_res = py_array_result.request();

  const double *parameters[2] = {&theta, a.begin()};
  double *jacobians[2] = {static_cast<double *>(buf_jac.ptr), nullptr};

  ceres::AutoDiffCostFunction<DiffRotorHepGAFunctor, 3, 1, 3>(
      new DiffRotorHepGAFunctor())
      .Evaluate(parameters, static_cast<double *>(buf_res.ptr), jacobians);

  py::list list;
  list.append(py_array_result);
  list.append(py_array_jac);

  return list;
}
struct DiffRotorGaalopFunctor {
  template <typename T>
  bool operator()(const T *th, const T *a, T *b) const {
    b[0] = (-(a[0] * sin(th[0] / 2.0) * sin(th[0] / 2.0))) -
           2.0 * a[1] * cos(th[0] / 2.0) * sin(th[0] / 2.0) +
           a[0] * cos(th[0] / 2.0) * cos(th[0] / 2.0); // e1
    b[1] = (-(a[1] * sin(th[0] / 2.0) * sin(th[0] / 2.0))) +
           2.0 * a[0] * cos(th[0] / 2.0) * sin(th[0] / 2.0) +
           a[1] * cos(th[0] / 2.0) * cos(th[0] / 2.0); // e2
    b[2] = a[2] * sin(th[0] / 2.0) * sin(th[0] / 2.0) +
           a[2] * cos(th[0] / 2.0) * cos(th[0] / 2.0); // e3
    return true;
  }
};

py::list CeresDiffRotorGaalop(const double theta, const Vec &a) {
  auto py_array_jac = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto py_array_result = py::array(py::buffer_info(
      nullptr, sizeof(double), py::format_descriptor<double>::value(), 2,
      {3, 1}, {sizeof(double), sizeof(double)}));

  auto buf_jac = py_array_jac.request();
  auto buf_res = py_array_result.request();

  const double *parameters[2] = {&theta, a.begin()};
  double *jacobians[2] = {static_cast<double *>(buf_jac.ptr), nullptr};

  ceres::AutoDiffCostFunction<DiffRotorGaalopFunctor, 3, 1, 3>(
      new DiffRotorGaalopFunctor())
      .Evaluate(parameters, static_cast<double *>(buf_res.ptr), jacobians);

  py::list list;
  list.append(py_array_result);
  list.append(py_array_jac);

  return list;
}

PYBIND11_PLUGIN(autodiff_multivector) {
  py::module m("autodiff_multivector", "autodiff_multivector");
  m.def("diff_adept", &AdeptDiffInnerProductVectorBivector);
  m.def("diff_ceres", &CeresDiffInnerProductVectorBivector);
  m.def("diff_ceres2", &CeresDiffInnerProductVectorVector);
  m.def("diff_ceres_rotor_versor", &CeresDiffRotorVersor);
  m.def("diff_adept_rotor_versor_forward", &AdeptDiffRotorVersorForward);
  m.def("diff_adept_rotor_versor_reverse", &AdeptDiffRotorVersorReverse);
  m.def("diff_ceres_rotor_matrix", &CeresDiffRotorMatrix);
  m.def("diff_adept_rotor_matrix_forward", &AdeptDiffRotorMatrixForward);
  m.def("diff_adept_rotor_matrix_reverse", &AdeptDiffRotorMatrixReverse);
  m.def("diff_ceres_rotor_hepga", &CeresDiffRotorHepGA);
  m.def("diff_ceres_rotor_gaalop", &CeresDiffRotorGaalop);
  m.def("diff_adept_rotor_hepga_forward", &AdeptDiffRotorHepGAForward);
  m.def("diff_adept_rotor_hepga_reverse", &AdeptDiffRotorHepGAReverse);
  return m.ptr();
}
