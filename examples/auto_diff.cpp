// GAME - Geometric Algebra Multivector Estimation
//
// Copyright (c) 2015, Norwegian University of Science and Technology
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of GAME nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVE CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "ceres/internal/autodiff.h"
#include "ceres/internal/eigen.h"
#include "ceres/autodiff_cost_function.h"
#include "ceres/cost_function.h"
#include <glog/logging.h>

#include <adept.h>

#include <iostream>
#include <sstream>
#include <string>

namespace py = pybind11;

struct F1 {
  template <typename T>
  bool operator()(const T *x, T *res) const {
    T tmp1 = x[0] * x[0];
    T tmp2 = sin(tmp1);
    T tmp3 = x[1] * tmp2;
    res[0] = tmp3;
    return true;
  }
};

struct F2 {
  template <typename T>
  bool operator()(const T *x, T *res) const {
    T tmp1 = x[0] * x[1];
    T tmp2 = sin(x[0]);
    T tmp3 = tmp1 + tmp2;
    res[0] =  tmp3;
    return true;
  }
};

template <typename Function>
py::list CeresAutoDiff(double x1, double x2) {
  auto array = py::array(py::buffer_info(nullptr, sizeof(double),
                                         py::format_descriptor<double>::value(),
                                         1, {2}, {sizeof(double)}));

  auto buf = array.request();

  double result;
  double x1x2[2] = {x1, x2};
  const double *parameters[1] = {&x1x2[0]};
  double *jacobians[1] = {reinterpret_cast<double *>(buf.ptr)};

  ceres::AutoDiffCostFunction<Function, 1, 2>(new Function())
      .Evaluate(parameters, &result, jacobians);

  py::list list;
  list.append(py::float_(result));
  list.append(array);

  return list;
}

template <typename Function>
py::list AdeptAutoDiff(double x1, double x2) {
  auto array = py::array(py::buffer_info(nullptr, sizeof(double),
                                         py::format_descriptor<double>::value(),
                                         1, {2}, {sizeof(double)}));

  auto buf = array.request();

  adept::Stack stack;
  double jac[2];
  std::array<adept::adouble, 2> ax{x1, x2};
  stack.new_recording();
  Function f;
  adept::adouble result;
  f(ax.begin(), &result);

  stack.independent(ax.begin(), 2);
  stack.dependent(result);

  stack.jacobian_reverse(reinterpret_cast<double *>(buf.ptr), true);
  // stack.jacobian_forward(static_cast<double *>(buf.ptr), true);

  std::stringstream ss;
  stack.print_statements(ss);

  py::list list;
  list.append(py::float_(result.value()));
  list.append(array);
  list.append(py::str(ss.str().c_str()));

  return list;
}

PYBIND11_PLUGIN(auto_diff) {
  py::module m("auto_diff", "auto_diff");
  m.def("f1", &AdeptAutoDiff<F1>);
  m.def("f2", &AdeptAutoDiff<F2>);
  m.def("f1_ceres", &CeresAutoDiff<F1>);
  m.def("f2_ceres", &CeresAutoDiff<F2>);

  return m.ptr();
}
