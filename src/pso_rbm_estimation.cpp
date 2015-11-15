//
// Created by lars on 13.11.15.
//

#include <ceres/ceres.h>
#include <glog/logging.h>

#include <boost/python.hpp>
#include <boost/numpy.hpp>

#include "game/types.h"
#include "game/ceres_python_utils.h"

namespace bp = boost::python;
namespace np = boost::numpy;

namespace game {

np::ndarray RotorRotatePoint(const np::ndarray& motor,
                             const np::ndarray& vector) {
  CheckContiguousArray(motor, "motor");
  CheckArrayShape(motor, "rotor", 8, 1);
  CheckContiguousArray(vector, "vector");
  CheckArrayShape(vector, "vector", 3, 1);

  double* motor_data = reinterpret_cast<double*>(motor.get_data());
  double* vector_data = reinterpret_cast<double*>(vector.get_data());

  versor::Vectord v;
  std::copy(vector_data, vector_data + 3, const_cast<double*>(v.begin()));
  versor::Motord m;
  std::copy(motor_data, motor_data + 8, const_cast<double*>(m.begin()));

  auto v2 = v.null().sp(m);

  np::ndarray result =
      np::zeros(bp::make_tuple(3, 1), np::dtype::get_builtin<double>());
  std::copy(v2.begin(), v2.begin() + 3,
            reinterpret_cast<double*>(result.get_data()));

  return result;
}



}

BOOST_PYTHON_MODULE(libpso) {
  np::initialize();
  bp::def("rotate_point", &game::RotorRotatePoint);
}
