#include <pybind11/pybind11.h>

#include "game/vsr/cga_op.h"
#include "game/vsr/cga_types.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddBivector(py::module &m) {
  py::class_<Biv>(m, "Biv")
      .def(py::init<double, double, double>())
      .def("__getitem__", &Biv::at)
      .def("__setitem__", [](Biv &arg, int idx, double val) { arg[idx] = val; })
      .def("norm", &Biv::norm)
      .def("rnorm", &Biv::rnorm)
      .def("duale", &Biv::duale)
      .def("unduale", &Biv::unduale)
      .def("unit", &Biv::unit)
      .def("rev", &Biv::reverse)
      .def("inv", &Biv::inverse)
      .def("spin", (Biv (Biv::*)(const Rot &) const) & Biv::spin)
      .def("exp", [](const Biv &biv) { return Gen::rotor(biv); })
      .def("__le__", [](const Biv &lhs, const Tri &rhs) { return lhs <= rhs; })
      .def("__le__",
           [](const Biv &lhs, const Biv &rhs) { return (lhs <= rhs)[0]; })
      .def("__xor__", [](const Biv &lhs, const Vec &rhs) { return lhs ^ rhs; })
      .def("__add__",
           [](const Biv &lhs, const Drv &rhs) { return Dll(lhs + rhs); })
      .def("__sub__", [](const Biv &lhs, const Biv &rhs) { return lhs - rhs; })
      .def("__add__", [](const Biv &lhs, const Biv &rhs) { return lhs + rhs; })
      .def("__mul__", [](const Biv &lhs, const Vec &rhs) { return lhs * rhs; })
      .def("__mul__", [](const Biv &lhs, const Biv &rhs) { return lhs * rhs; })
      .def("__mul__", [](const Biv &lhs, double rhs) { return lhs * rhs; })
      .def("__mul__", [](const Biv &lhs, const Rot &rhs) { return lhs * rhs; })
      .def("__div__", [](const Biv &lhs, double rhs) { return lhs / rhs; })
      .def("comm", [](const Biv &lhs,
                      const Rot &rhs) { return (lhs * rhs - rhs * lhs) * 0.5; })
      .def("acomm",
           [](const Biv &lhs, const Rot &rhs) {
             return (lhs * rhs + rhs * lhs) * 0.5;
           })
      .def("comm",
           [](const Biv &lhs, const Vec &rhs) {
             return Vec(lhs * rhs - rhs * lhs) * 0.5;
           })
      .def("comm",
           [](const Biv &lhs, const Biv &rhs) {
             return Biv(lhs * rhs - rhs * lhs) * 0.5;
           })
      .def("acomm",
           [](const Biv &lhs, const Vec &rhs) {
             return Vec(lhs * rhs + rhs * lhs) * 0.5;
           })
      .def("__repr__",
           [](const Biv &arg) {
             std::stringstream ss;
             ss.precision(4);
             ss << "Biv: [";
             for (int i = 0; i < arg.Num; ++i) {
               ss << " " << arg[i];
             }
             ss << " ]";
             return ss.str();
           })
      .def_buffer([](Biv &arg) -> py::buffer_info {
        return py::buffer_info(
            arg.data(), sizeof(double), py::format_descriptor<double>::format(),
            1, {static_cast<unsigned long>(arg.Num)}, {sizeof(double)});
      });
}

} // namespace python

} // namespace vsr
