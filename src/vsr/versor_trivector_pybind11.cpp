#include <pybind11/pybind11.h>

#include "game/vsr/cga_op.h"
#include "game/vsr/cga_types.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddTrivector(py::module &m) {
  py::class_<Tri>(m, "Tri")
      .def(py::init<double>())
      .def("__getitem__", &Tri::at)
      .def("norm", &Tri::norm)
      .def("rnorm", &Tri::rnorm)
      .def("unit", &Tri::unit)
      .def("rev", &Tri::reverse)
      .def("inv", &Tri::inverse)
      .def("duale", &Tri::duale)
      .def("unduale", &Tri::unduale)
      .def("trs", [](const Tri &arg) { return Gen::trs(arg); })
      .def("__neg__", [](const Tri &arg) { return -arg; })
      .def("__xor__", [](const Tri &lhs, const Tri &rhs) { return lhs ^ rhs; })
      .def("__mul__", [](const Tri &lhs, double rhs) { return lhs * rhs; })
      .def("__mul__", [](const Tri &lhs, const Tri &rhs) { return lhs * rhs; })
      .def("__mul__", [](const Tri &lhs, const Biv &rhs) { return lhs * rhs; })
      .def("__sub__", [](const Tri &lhs, const Tri &rhs) { return lhs - rhs; })
      .def("__add__", [](const Tri &lhs, const Tri &rhs) { return lhs + rhs; })
      .def("spin", (Tri (Tri::*)(const Rot &) const) & Tri::spin)
      .def("null", &Tri::null)
      .def("__repr__",
           [](const Tri &arg) {
             std::stringstream ss;
             ss.precision(2);
             ss << "Tri: [";
             for (int i = 0; i < arg.Num; ++i) {
               ss << " " << arg[i];
             }
             ss << " ]";
             return ss.str();
           })
      .def_buffer([](Tri &arg) -> py::buffer_info {
        return py::buffer_info(
            arg.data(), sizeof(double), py::format_descriptor<double>::format(),
            1, {static_cast<unsigned long>(arg.Num)}, {sizeof(double)});
      });
}

} // namespace python

} // namespace vsr
