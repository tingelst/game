#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ceres/internal/autodiff.h"
#include "ceres/internal/eigen.h"
#include "ceres/rotation.h"

#include <glog/logging.h>

#include <game/vsr/cga_op.h>
#include <game/motor_parameterization.h>

#include <iostream>

using namespace game;
using namespace vsr::cga;
namespace py = pybind11;

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
  m.def("jacobian_exp", &ExpJacobian).def("jacobian_polar", &PolarJacobian);

  return m.ptr();
}
