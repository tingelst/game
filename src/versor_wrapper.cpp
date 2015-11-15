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

  np::ndarray Translator(const np::ndarray& translation) {
    CheckContiguousArrayAndArrayShape(translation, "translation", 3, 1);

    double* translation_data =
        reinterpret_cast<double*>(translation.get_data());

    versor::Vectord t = versor::Vectord{
        translation_data[0], translation_data[1], translation_data[2]};

    versor::Translatord trs{1.0, -0.5 * t[0], -0.5 * t[1], -0.5 * t[2]};

    np::ndarray result =
        np::zeros(bp::make_tuple(4, 1), np::dtype::get_builtin<double>());
    std::copy(trs.begin(), trs.begin() + 4,
              reinterpret_cast<double*>(result.get_data()));
    return result;
  }

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


  game::versor::Vectord(game::versor::Vectord::*spin1)(const game::versor::Rotord&) const =
      &game::versor::Vectord::spin;

BOOST_PYTHON_MODULE(libversor) {
  np::initialize();
  bp::class_<game::Versor>("Versor")
      .def("translator", &game::Versor::Translator)
      .def("dual_line", &game::Versor::DualLine)
      .def("motor_transform_line", &game::Versor::MotorTransformLine);

  bp::class_<game::versor::Scalard>("Scalar")
      .def("__getitem__", &game::versor::Scalard::at);

  bp::class_<game::versor::Pointd>("Point")
      .def(bp::init<game::versor::RotorPoint>())
      .def("__getitem__", &game::versor::Pointd::at)
      .def(bp::other<game::versor::Rotord>() * bp::self)
      .def(bp::self | bp::other<game::versor::Rotord>())
      .def(bp::self | bp::other<game::versor::Motord>())
      .add_property("num", &game::versor::Pointd::get_num_bases)
      //.def("spin", &game::versor::Pointd::spin2)
      ;

  bp::class_<game::versor::DualLined>(
      "DualLine", bp::init<double, double, double, double, double, double>())
      .def("__getitem__", &game::versor::DualLined::at)
      .def(bp::self | bp::other<game::versor::Rotord>())
      .def(bp::self | bp::other<game::versor::Motord>())
      .def("__str__", &game::versor::DualLined::to_string);
  //.def("spin", &game::versor::Pointd::spin2)
  ;

  bp::class_<game::versor::Bivectord>("Bivector",
                                      bp::init<double, double, double>())
      .def("__getitem__", &game::versor::Bivectord::at)
      .def("unit", &game::versor::Bivectord::unit)
      .def("norm", &game::versor::Bivectord::norm)
      .def("print", &game::versor::Bivectord::print)
      .def("reverse", &game::versor::Bivectord::reverse)
      .def("dual", &game::versor::Bivectord::duale)
      //.def("spin", &game::versor::Bivectord::spin)
      .def(bp::self + bp::self);


  bp::class_<game::versor::Vectord>("Vector",
                                    bp::init<double, double, double>())
      .def(bp::init<game::versor::Vectord>())
      .def("__getitem__", &game::versor::Vectord::at)
      .def("unit", &game::versor::Vectord::unit)
      .def("norm", &game::versor::Vectord::norm)
      .def("print", &game::versor::Vectord::print)
      .def("reverse", &game::versor::Vectord::reverse)
      .def("inverse", &game::versor::Vectord::inverse)
      .def("dual", &game::versor::Vectord::duale)
      .def("null", &game::versor::Vectord::null)
      .def("spin", spin1)
      .add_property("num", &game::versor::Vectord::get_num_bases)
      .def(bp::self | bp::other<game::versor::Rotord>())
      .def(bp::self <= bp::self)
      .def(bp::self + bp::self)
      .def(bp::self ^ bp::self)
      .def(bp::self * bp::self)
      .def(bp::self / bp::self)
      .def("__str__", &game::versor::Vectord::to_string);

  bp::class_<game::versor::Motord>(
      "Motor", bp::init<double, double, double, double, double, double, double,
                        double>())
      .def("__str__", &game::versor::Motord::to_string);

  bp::class_<game::versor::RotorPoint>("RotorPoint")
      .def(bp::self * bp::other<game::versor::Rotord>());

  bp::class_<game::versor::VecEucd>("VecEuc")
      .def(bp::self * bp::other<game::versor::Rotord>())
      .def("__getitem__", &game::versor::VecEucd::at);

  bp::class_<game::versor::Rotord>("Rotor",
                                   bp::init<double, double, double, double>())
      .def("__getitem__", &game::versor::Rotord::at)
      .def("unit", &game::versor::Rotord::unit)
      .def("norm", &game::versor::Rotord::norm)
      .def("print", &game::versor::Rotord::print)
      .def("reverse", &game::versor::Rotord::reverse)
      .add_property("num", &game::versor::Rotord::get_num_bases)
      .def(bp::self * bp::other<game::versor::Vectord>())
      .def(bp::other<game::versor::Vectord>() * bp::self);
}
