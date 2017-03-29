#include <pybind11/pybind11.h>

#include "game/vsr/cga_op.h"
#include "game/vsr/cga_types.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddDualPlane(py::module &m) {
  py::class_<Dlp>(m, "Dlp")
      .def(py::init<double, double, double, double>())
      .def("__init__",
           [](Dlp &instance, const Vec &arg, double distance) {
             new (&instance) Dlp(arg[0], arg[1], arg[2], distance);
           })
      .def("duale", &Dlp::duale)
      .def("unduale", &Dlp::unduale)
      .def("dual", &Dlp::dual)
      .def("undual", &Dlp::undual)
      .def("unit", &Dlp::unit)
      .def("rev", &Dlp::reverse)
      .def("inv", &Dlp::inverse)
      .def("loc",
           [](const Dlp &arg, const Pnt &arg2) {
             return Flat::location(arg, arg2, true);
           })
      .def("dir",
           [](const Dlp &arg) {
             Dlp dlp = arg.unit();
             return Drv(dlp[0], dlp[1], dlp[2]);
           })
      .def("vec", [](const Dlp &arg) { return Vec(arg[0], arg[1], arg[2]); })
      .def("spin", (Dlp (Dlp::*)(const Mot &) const) & Dlp::spin)
    .def("comm",
         [](const Dlp &lhs, const Dlp &rhs) {
           return Dll(lhs * rhs);
         })
      .def("__mul__",
           [](const Dlp &lhs, const Dlp &rhs) { return Mot(lhs * rhs); })
      .def("__sub__",
           [](const Dlp &lhs, const Dlp &rhs) { return Mot(lhs - rhs); })
      .def("__mul__", [](const Dlp &lhs, double rhs) { return lhs * rhs; })
      .def("__div__", [](const Dlp &lhs, double rhs) { return lhs / rhs; })
      .def("__leq__", [](const Dlp &lhs, const Dlp &rhs) { return lhs <= rhs; })
      .def("lc", [](const Dlp &lhs, const Dlp &rhs) { return (lhs <= rhs)[0]; })
      .def("__repr__",
           [](const Dlp &arg) {
             std::stringstream ss;
             ss.precision(4);
             ss << "Dlp: [";
             for (int i = 0; i < arg.Num; ++i) {
               ss << " " << arg[i];
             }
             ss << " ]";
             return ss.str();
           })
      .def_buffer([](Dlp &arg) -> py::buffer_info {
        return py::buffer_info(
            arg.data(), sizeof(double), py::format_descriptor<double>::format(),
            1, {static_cast<unsigned long>(arg.Num)}, {sizeof(double)});
      });
}

} // namespace python

} // namespace vsr
