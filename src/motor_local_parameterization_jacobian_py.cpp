#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ceres/autodiff_cost_function.h"
#include "ceres/autodiff_local_parameterization.h"
#include "ceres/cost_function.h"
#include "ceres/internal/autodiff.h"
#include "ceres/internal/eigen.h"
#include "ceres/rotation.h"

#include <glog/logging.h>

#include <game/vsr/cga_op.h>
#include <game/vsr/generic_op.h>

#include <game/motor_parameterization.h>
#include <iostream>

using namespace game;
using namespace vsr::cga;
namespace py = pybind11;

struct RotorPlus {
  template <typename T>
  bool operator()(const T *x, const T *delta, T *x_plus_delta) const {
    const T squared_norm_delta =
        delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
    T q_delta[4];
    if (squared_norm_delta > T(0.0)) {
      T norm_delta = sqrt(squared_norm_delta);
      const T sin_delta_by_delta = sin(norm_delta) / norm_delta;
      q_delta[0] = cos(norm_delta);
      q_delta[1] = sin_delta_by_delta * delta[0];
      q_delta[2] = sin_delta_by_delta * delta[1];
      q_delta[3] = sin_delta_by_delta * delta[2];
    } else {
      // We do not just use q_delta = [1,0,0,0] here because that is a
      // constant and when used for automatic differentiation will
      // lead to a zero derivative. Instead we take a first order
      // approximation and evaluate it at zero.
      q_delta[0] = T(1.0);
      q_delta[1] = delta[0];
      q_delta[2] = delta[1];
      q_delta[3] = delta[2];
    }

    Rotor<T> r = Rotor<T>{q_delta[0], -q_delta[1], -q_delta[2], -q_delta[3]} *
                 Rotor<T>{x[0], x[1], x[2], x[3]};

    x_plus_delta[0] = r[0];
    x_plus_delta[1] = r[1];
    x_plus_delta[2] = r[2];
    x_plus_delta[3] = r[3];

    return true;
  }
};

struct VectorCorrespondencesCostFunctor {
  VectorCorrespondencesCostFunctor(const Vec &a, const Vec &b) : a_(a), b_(b) {}

  template <typename T>
  auto operator()(const T *const rotor, T *residual) const -> bool {
    Rotor<T> R(rotor);
    Vector<T> a(a_);
    Vector<T> b(b_);
    Vector<T> c = a.spin(R);

    for (int i = 0; i < 3; ++i) {
      residual[i] = c[i] - b[i];
    }

    return true;
  }

 private:
  const Vec a_;
  const Vec b_;
};

struct LineCorrespondencesCostFunctor {
  LineCorrespondencesCostFunctor(const Dll &a, const Dll &b) : a_(a), b_(b) {}

  template <typename T>
  auto operator()(const T *const motor, T *residual) const -> bool {
    Motor<T> M(motor);
    DualLine<T> a(a_);
    DualLine<T> b(b_);
    DualLine<T> c = a.spin(M);

    for (int i = 0; i < 6; ++i) {
      residual[i] = c[i] - b[i];
    }

    return true;
  }

 private:
  const Dll a_;
  const Dll b_;
};

struct LineCommutatorCostFunctor {
  LineCommutatorCostFunctor(const Dll &a, const Dll &b) : a_(a), b_(b) {}

  template <typename T>
  auto operator()(const T *const motor, T *residual) const -> bool {
    Motor<T> M(motor);
    DualLine<T> a(a_);
    DualLine<T> b(b_);
    DualLine<T> c = a.spin(M);
    DualLine<T> comm = (c * b - b * c) * Scalar<T>(0.5);

    for (int i = 0; i < 6; ++i) {
      residual[i] = comm[i];
    }

    return true;
  }

 private:
  const Dll a_;
  const Dll b_;
};

struct PointCorrespondencesCostFunctor {
  PointCorrespondencesCostFunctor(const Pnt &a, const Pnt &b) : a_(a), b_(b) {}

  template <typename T>
  auto operator()(const T *const motor, T *residual) const -> bool {
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

namespace {
template <typename ScalarType, int Rows, int Cols>
py::array_t<ScalarType> Matrix() {
  return py::array(py::buffer_info(
      nullptr, /* Pointer to data (nullptr -> ask NumPy to allocate!) */
      sizeof(ScalarType),                         /* Size of one item */
      py::format_descriptor<ScalarType>::value(), /* Buffer format */
      2,                                          /* How many dimensions? */
      {Rows, Cols}, /* Number of elements for each dimension */
      {sizeof(double) * Rows, sizeof(double)} /* Strides for each dimension */
      ));
}
}

py::array_t<double> AnalyticDiffRotorCost(const Rot &rot, const Vec &a,
                                          const Vec &b) {
  // auto result = py::array(py::buffer_info(
  //     nullptr,        /* Pointer to data (nullptr -> ask NumPy to allocate!)
  //     */
  //     sizeof(double), /* Size of one item */
  //     py::format_descriptor<double>::value(), /* Buffer format */
  //     2,                                      /* How many dimensions? */
  //     {3, 4}, /* Number of elements for each dimension */
  //     {sizeof(double) * 3, sizeof(double)} /* Strides for each dimension */
  //     ));

  auto result = ::Matrix<double, 3, 4>();

  auto buf = result.request();

  auto row1 =
      (~(a * Vec(1.0, 0.0, 0.0)) * rot - rot * ~(a * Vec(1.0, 0.0, 0.0)));
  // +((Vec(1.0, 0.0, 0.0) * a) * ~rot + ~rot * (Vec(1.0, 0.0, 0.0) * a));

  auto row2 =
      (~(-Vec(0.0, 1.0, 0.0) * a) * ~rot - ~rot * ~(-Vec(0.0, 1.0, 0.0) * a));
  // ((Vec(0.0, 1.0, 0.0) * a) * ~rot + ~rot * (Vec(0.0, 1.0, 0.0) * a));

  auto row3 =
      (~(-Vec(0.0, 0.0, 1.0) * a) * ~rot - ~rot * ~(-Vec(0.0, 0.0, 1.0) * a));
  // ((Vec(0.0, 0.0, 1.0) * a) * ~rot + ~rot * (Vec(0.0, 0.0, 1.0) * a));

  auto jac = reinterpret_cast<double *>(buf.ptr);
  for (int j = 0; j < 4; ++j) {
    jac[j] = row1[j];
    jac[4 + j] = row2[j];
    jac[8 + j] = row3[j];
  }

  return result;
}

py::list DiffRotorCost(const Rot &rot, const Vec &a, const Vec &b) {
  auto result = py::array(py::buffer_info(
      nullptr,        /* Pointer to data (nullptr -> ask NumPy to allocate!) */
      sizeof(double), /* Size of one item */
      py::format_descriptor<double>::value(), /* Buffer format */
      2,                                      /* How many dimensions? */
      {3, 4}, /* Number of elements for each dimension */
      {sizeof(double) * 3, sizeof(double)} /* Strides for each dimension */
      ));

  auto buf = result.request();

  auto np_residual = py::array(py::buffer_info(
      nullptr,        /* Pointer to data (nullptr -> ask NumPy to allocate!) */
      sizeof(double), /* Size of one item */
      py::format_descriptor<double>::value(), /* Buffer format */
      1,                                      /* How many dimensions? */
      {3},             /* Number of elements for each dimension */
      {sizeof(double)} /* Strides for each dimension */
      ));

  auto np_residual_buf = np_residual.request();

  const double *parameters[1] = {rot.begin()};
  double *jacobian_array[1] = {static_cast<double *>(buf.ptr)};

  ceres::AutoDiffCostFunction<VectorCorrespondencesCostFunctor, 3, 4>(
      new VectorCorrespondencesCostFunctor(a, b))
      .Evaluate(parameters, static_cast<double *>(np_residual_buf.ptr),
                jacobian_array);

  py::list list;
  list.append(np_residual);
  list.append(result);

  return list;
}

py::array_t<double> DiffRotorLocalParameterization(const Rot &rot) {
  auto result = py::array(py::buffer_info(
      nullptr,        /* Pointer to data (nullptr -> ask NumPy to allocate!) */
      sizeof(double), /* Size of one item */
      py::format_descriptor<double>::value(), /* Buffer format */
      2,                                      /* How many dimensions? */
      {4, 3}, /* Number of elements for each dimension */
      {sizeof(double) * 4, sizeof(double)} /* Strides for each dimension */
      ));

  auto buf = result.request();

  ceres::AutoDiffLocalParameterization<RotorPlus, 4, 3>(new RotorPlus())
      .ComputeJacobian(rot.begin(), reinterpret_cast<double *>(buf.ptr));

  return result;
}

py::array_t<double> DiffCostLinesComm(const Mot &mot, const Dll &a,
                                      const Dll &b) {
  auto result = py::array(py::buffer_info(
      nullptr,        /* Pointer to data (nullptr -> ask NumPy to allocate!) */
      sizeof(double), /* Size of one item */
      py::format_descriptor<double>::value(), /* Buffer format */
      2,                                      /* How many dimensions? */
      {6, 8},                 /* Number of elements for each dimension */
      {sizeof(double) * 6, 8} /* Strides for each dimension */
      ));

  auto buf = result.request();

  const double *parameters[1] = {mot.begin()};
  double *jacobian_array[1] = {static_cast<double *>(buf.ptr)};

  double residuals[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  ceres::AutoDiffCostFunction<LineCommutatorCostFunctor, 6, 8>(
      new LineCommutatorCostFunctor(a, b))
      .Evaluate(parameters, residuals, jacobian_array);

  return result;
}
py::array_t<double> DiffCostLines(const Mot &mot, const Dll &a, const Dll &b) {
  auto result = py::array(py::buffer_info(
      nullptr,        /* Pointer to data (nullptr -> ask NumPy to allocate!) */
      sizeof(double), /* Size of one item */
      py::format_descriptor<double>::value(), /* Buffer format */
      2,                                      /* How many dimensions? */
      {6, 8},                 /* Number of elements for each dimension */
      {sizeof(double) * 6, 8} /* Strides for each dimension */
      ));

  auto buf = result.request();

  const double *parameters[1] = {mot.begin()};
  double *jacobian_array[1] = {static_cast<double *>(buf.ptr)};

  double residuals[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  ceres::AutoDiffCostFunction<LineCorrespondencesCostFunctor, 6, 8>(
      new LineCorrespondencesCostFunctor(a, b))
      .Evaluate(parameters, residuals, jacobian_array);

  return result;
}

py::array_t<double> DiffCost(const Mot &mot, const Pnt &a, const Pnt &b) {
  auto result = py::array(py::buffer_info(
      nullptr,        /* Pointer to data (nullptr -> ask NumPy to allocate!) */
      sizeof(double), /* Size of one item */
      py::format_descriptor<double>::value(), /* Buffer format */
      2,                                      /* How many dimensions? */
      {3, 8},                 /* Number of elements for each dimension */
      {sizeof(double) * 3, 8} /* Strides for each dimension */
      ));

  auto buf = result.request();

  const double *parameters[1] = {mot.begin()};
  double *jacobian_array[1] = {static_cast<double *>(buf.ptr)};

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
  bool operator()(const T *m, const T *x, T *x_prime) const {
    Motor<T> M(m);
    Point<T> p(x);

    Point<T> p_prime = p.spin(M);

    for (int i = 0; i < p.Num; ++i) x_prime[i] = p_prime[i];

    return true;
  }
};

py::array_t<double> DiffPoint(const Mot &mot, const Pnt &p) {
  auto result = py::array(py::buffer_info(
      nullptr,        /* Pointer to data (nullptr -> ask NumPy to allocate!) */
      sizeof(double), /* Size of one item */
      py::format_descriptor<double>::value(), /* Buffer format */
      2,                                      /* How many dimensions? */
      {5, 8},                 /* Number of elements for each dimension */
      {sizeof(double) * 5, 8} /* Strides for each dimension */
      ));

  auto buf = result.request();

  const double *parameters[2] = {mot.begin(), p.begin()};
  double *jacobian_array[2] = {static_cast<double *>(buf.ptr), nullptr};

  double x_plus_delta[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  // Autodiff jacobian at delta_x = 0.
  ceres::internal::AutoDiff<TransformPoint, double, 5, 8>::Differentiate(
      TransformPoint(), parameters, 5, x_plus_delta, jacobian_array);

  return result;
}

py::array_t<double> PolarJacobian(const Mot &mot1, const Mot &mot2) {
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
  const double *parameters[2] = {mot1.begin(), mot2.begin()};
  double *jacobian_array[2] = {NULL, static_cast<double *>(buf.ptr)};

  double x_plus_delta[kGlobalSize] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  // Autodiff jacobian at delta_x = 0.
  ceres::internal::AutoDiff<
      MotorPolarDecomposition, double, kGlobalSize,
      kLocalSize>::Differentiate(MotorPolarDecomposition(), parameters,
                                 kGlobalSize, x_plus_delta, jacobian_array);

  return result;
}

py::array_t<double> ExpJacobian(const Mot &mot, const Dll &dll) {
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
  const double *parameters[2] = {mot.begin(), dll.begin()};
  double *jacobian_array[2] = {NULL, static_cast<double *>(buf.ptr)};

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
      .def("diff_cost", &DiffCost)
      .def("diff_cost_lines", &DiffCostLines)
      .def("diff_cost_lines_comm", &DiffCostLinesComm)
      .def("diff_rotor_cost", &DiffRotorCost)
      .def("analytic_diff_rotor_cost", &AnalyticDiffRotorCost)
      .def("rotor_local_parameterization", &DiffRotorLocalParameterization);

  return m.ptr();
}
