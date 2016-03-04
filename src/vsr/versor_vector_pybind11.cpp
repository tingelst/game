#include <pybind11/pybind11.h>

#include "game/vsr/cga_op.h"
#include "game/vsr/cga_types.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddVector(py::module &m) {
  py::class_<Vec>(m, "Vec")
      .def(py::init<double, double, double>())
      .def("__getitem__", &Vec::at)
      .def("norm", &Vec::norm)
      .def("rnorm", &Vec::rnorm)
      .def("unit", &Vec::unit)
      .def("rev", &Vec::reverse)
      .def("inv", &Vec::inverse)
      .def("duale", &Vec::duale)
      .def("unduale", &Vec::unduale)
      .def("trs", [](const Vec &arg) { return Gen::trs(arg); })
      .def("__xor__", [](const Vec &lhs, const Vec &rhs) { return lhs ^ rhs; })
      .def("__mul__", [](const Vec &lhs, double rhs) { return lhs * rhs; })
      .def("__mul__", [](const Vec &lhs, const Vec &rhs) { return lhs * rhs; })
      .def("__sub__", [](const Vec &lhs, const Vec &rhs) { return lhs - rhs; })
      .def("__add__", [](const Vec &lhs, const Vec &rhs) { return lhs + rhs; })
      .def("__le__", [](const Vec &lhs, const Biv &rhs) { return lhs <= rhs; })
      .def("spin", (Vec (Vec::*)(const Rot &) const) & Vec::spin)
      .def("null", &Vec::null)
      .def("__repr__",
           [](const Vec &arg) {
             std::stringstream ss;
             ss.precision(2);
             ss << "Vec: [";
             for (int i = 0; i < arg.Num; ++i) {
               ss << " " << arg[i];
             }
             ss << " ]";
             return ss.str();
           })
      .def_buffer([](Vec &arg) -> py::buffer_info {
        return py::buffer_info(
            arg.data(), sizeof(double), py::format_descriptor<double>::value(),
            1, {static_cast<unsigned long>(arg.Num)}, {sizeof(double)});
      });
}

} // namespace python

} // namespace vsr
