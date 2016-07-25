#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <Eigen/Dense>
using Eigen::Matrix;
using Eigen::Dynamic;

#include "ceres/internal/autodiff.h"
#include "ceres/internal/eigen.h"
#include "ceres/autodiff_cost_function.h"
#include "ceres/cost_function.h"

#include <glog/logging.h>

#include <adept.h>

#include <game/vsr/cga_op.h>

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
  for (int i = 0; i < 3; ++i) res[i] = result[i];
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

PYBIND11_PLUGIN(autodiff_multivector) {
  py::module m("autodiff_multivector", "autodiff_multivector");
  m.def("diff_adept", &AdeptDiffInnerProductVectorBivector);
  m.def("diff_ceres", &CeresDiffInnerProductVectorBivector);
  m.def("diff_ceres2", &CeresDiffInnerProductVectorVector);
  return m.ptr();
}
