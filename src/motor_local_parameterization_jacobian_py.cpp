#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ceres/internal/autodiff.h"
#include "ceres/internal/eigen.h"
#include "ceres/rotation.h"
#include "ceres/autodiff_cost_function.h"
#include "ceres/cost_function.h"

#include <glog/logging.h>

#include <game/vsr/cga_op.h>
#include <game/motor_parameterization.h>

#include <iostream>

using namespace game;
using namespace vsr::cga;
namespace py = pybind11;

struct PointCorrespondencesCostFunctor {
  PointCorrespondencesCostFunctor(const Pnt& a, const Pnt& b) : a_(a), b_(b) {}

  template <typename T>
  auto operator()(const T* const motor, T* residual) const -> bool {
    Motor<T> M(motor);
    Point<T> a(a_);
    Point<T> b(b_);
    Point<T> c = a.spin(M);


    for (int i = 0; i < 3; ++i) {
      residual[i] = c[i] - b[i];
    }

    return true;
  }

 private:
  const Pnt a_;
  const Pnt b_;
};

py::array_t<double> DiffCost(const Mot& mot, const Pnt& a, const Pnt& b) {
  auto result = py::array(py::buffer_info(
      nullptr,        /* Pointer to data (nullptr -> ask NumPy to allocate!) */
      sizeof(double), /* Size of one item */
      py::format_descriptor<double>::value(), /* Buffer format */
      2,                                      /* How many dimensions? */
      {3, 8},                 /* Number of elements for each dimension */
      {sizeof(double) * 3, 8} /* Strides for each dimension */
      ));

  auto buf = result.request();

  const double* parameters[1] = {mot.begin()};
  double* jacobian_array[1] = {static_cast<double*>(buf.ptr)};

  double residuals[3] = {0.0, 0.0, 0.0};
  // Autodiff jacobian at delta_x = 0.
  // PointCorrespondencesCostFunctor functor(a, b);
  // ceres::internal::AutoDiff<PointCorrespondencesCostFunctor, double, 3,
  //                           8>::Differentiate(functor, parameters, 3,
  //                           residual,
  //                                             jacobian_array);

  ceres::AutoDiffCostFunction<PointCorrespondencesCostFunctor, 3, 8>(
      new PointCorrespondencesCostFunctor(a, b))
      .Evaluate(parameters, residuals, jacobian_array);

  std::cout << ceres::MatrixRef(residuals, 1, 3) << std::endl;

  return result;
}

struct TransformPoint {
  template <typename T>
  bool operator()(const T* m, const T* x, T* x_prime) const {
    Motor<T> M(m);
    Point<T> p(x);

    Point<T> p_prime = p.spin(M);

    for (int i = 0; i < p.Num; ++i) x_prime[i] = p_prime[i];

    return true;
  }
};

py::array_t<double> DiffPoint(const Mot& mot, const Pnt& p) {
  auto result = py::array(py::buffer_info(
      nullptr,        /* Pointer to data (nullptr -> ask NumPy to allocate!) */
      sizeof(double), /* Size of one item */
      py::format_descriptor<double>::value(), /* Buffer format */
      2,                                      /* How many dimensions? */
      {5, 8},                 /* Number of elements for each dimension */
      {sizeof(double) * 5, 8} /* Strides for each dimension */
      ));

  auto buf = result.request();

  const double* parameters[2] = {mot.begin(), p.begin()};
  double* jacobian_array[2] = {static_cast<double*>(buf.ptr), nullptr};

  double x_plus_delta[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  // Autodiff jacobian at delta_x = 0.
  ceres::internal::AutoDiff<TransformPoint, double, 5, 8>::Differentiate(
      TransformPoint(), parameters, 5, x_plus_delta, jacobian_array);

  return result;
}

py::array_t<double> PolarJacobian(const Mot& mot1, const Mot& mot2) {
  const int kGlobalSize = 8;
  const int kLocalSize = 8;

  auto result = py::array(py::buffer_info(
      nullptr,        /* Pointer to data (nullptr -> ask NumPy to allocate!) */
      sizeof(double), /* Size of one item */
      py::format_descriptor<double>::value(), /* Buffer format */
      2,                                      /* How many dimensions? */
      {kGlobalSize, kLocalSize}, /* Number of elements for each dimension */
      {sizeof(double) * kGlobalSize, kLocalSize}
      /* Strides for each dimension */
      ));

  auto buf = result.request();

  double zero_delta[kLocalSize] = {0.0, 0.0, 0.0, 0.0, 0.0};
  const double* parameters[2] = {mot1.begin(), mot2.begin()};
  double* jacobian_array[2] = {NULL, static_cast<double*>(buf.ptr)};

  double x_plus_delta[kGlobalSize] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  // Autodiff jacobian at delta_x = 0.
  ceres::internal::AutoDiff<
      MotorPolarDecomposition, double, kGlobalSize,
      kLocalSize>::Differentiate(MotorPolarDecomposition(), parameters,
                                 kGlobalSize, x_plus_delta, jacobian_array);

  return result;
}

py::array_t<double> ExpJacobian(const Mot& mot, const Dll& dll) {
  const int kGlobalSize = 8;
  const int kLocalSize = 6;

  auto result = py::array(py::buffer_info(
      nullptr,        /* Pointer to data (nullptr -> ask NumPy to allocate!) */
      sizeof(double), /* Size of one item */
      py::format_descriptor<double>::value(), /* Buffer format */
      2,                                      /* How many dimensions? */
      {kGlobalSize, kLocalSize}, /* Number of elements for each dimension */
      {sizeof(double) * kGlobalSize, kLocalSize}
      /* Strides for each dimension */
      ));

  auto buf = result.request();

  double zero_delta[kLocalSize] = {0.0, 0.0, 0.0, 0.0, 0.0};
  const double* parameters[2] = {mot.begin(), dll.begin()};
  double* jacobian_array[2] = {NULL, static_cast<double*>(buf.ptr)};

  double x_plus_delta[kGlobalSize] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  // Autodiff jacobian at delta_x = 0.
  ceres::internal::AutoDiff<
      MotorFromBivectorGenerator, double, kGlobalSize,
      kLocalSize>::Differentiate(MotorFromBivectorGenerator(), parameters,
                                 kGlobalSize, x_plus_delta, jacobian_array);

  return result;
}

PYBIND11_PLUGIN(motor_jacobian) {
  py::module m("motor_jacobian", "motor_jacobian");
  m.def("jacobian_exp", &ExpJacobian)
      .def("jacobian_polar", &PolarJacobian)
      .def("diff_point", &DiffPoint)
      .def("diff_cost", &DiffCost);

  return m.ptr();
}
