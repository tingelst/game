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

struct Versor {
  Versor() {}
  Versor(const Versor& versor) {}

  np::ndarray DualLine(const np::ndarray& point, const np::ndarray& direction) {
    CheckContiguousArray(point, "point");
    CheckArrayShape(point, "point", 3, 1);
    CheckContiguousArray(direction, "direction");
    CheckArrayShape(direction, "direction", 3, 1);

    double* point_data = reinterpret_cast<double*>(point.get_data());
    double* direction_data = reinterpret_cast<double*>(direction.get_data());

    versor::Pointd p =
        versor::Vectord{point_data[0], point_data[1], point_data[2]}.null();
    versor::Vectord u = versor::Vectord{direction_data[0], direction_data[1],
                                        direction_data[2]};
    auto l = p ^ u ^ versor::Infd{1.0};
    auto dll = l.dual();

    np::ndarray result =
        np::zeros(bp::make_tuple(6, 1), np::dtype::get_builtin<double>());
    std::copy(dll.begin(), dll.begin() + 6,
              reinterpret_cast<double*>(result.get_data()));
    return result;
  }

  np::ndarray MotorTransformLine(const np::ndarray& motor,
                                 const np::ndarray& line) {
    CheckContiguousArrayAndArrayShape(motor, "motor", 8, 1);
    CheckContiguousArrayAndArrayShape(line, "line", 6, 1);

    double* motor_data = reinterpret_cast<double*>(motor.get_data());
    double* line_data = reinterpret_cast<double*>(line.get_data());

    versor::Motord m{motor_data[0], motor_data[1], motor_data[2],
                     motor_data[3], motor_data[4], motor_data[5],
                     motor_data[6], motor_data[7]};

    versor::DualLined dll1{line_data[0], line_data[1], line_data[2],
                           line_data[3], line_data[4], line_data[5]};

    auto dll2 = dll1.spin(m);

    np::ndarray result =
        np::zeros(bp::make_tuple(6, 1), np::dtype::get_builtin<double>());
    std::copy(dll2.begin(), dll2.begin() + 6,
              reinterpret_cast<double*>(result.get_data()));
    return result;
  }
};

}

BOOST_PYTHON_MODULE(libversor) {
  np::initialize();
  bp::class_<game::Versor>("Versor")
      .def("dual_line", &game::Versor::DualLine)
      .def("motor_transform_line", &game::Versor::MotorTransformLine);
}
