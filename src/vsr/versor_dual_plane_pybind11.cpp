#include <pybind11/pybind11.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/cga_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddDualPlane(py::module &m) {
  py::class_<Dlp>(m, "Dlp")
      .def(py::init<double, double, double, double>())
      .def("duale", &Dlp::duale)
      .def("unduale", &Dlp::unduale)
      .def("dual", &Dlp::dual)
      .def("undual", &Dlp::undual)
      .def("unit", &Dlp::unit)
      .def("rev", &Dlp::reverse)
      .def("inv", &Dlp::inverse)
      .def("spin", (Dlp (Dlp::*)(const Mot &) const) & Dlp::spin)
      .def("__mul__", [](const Dlp &lhs, const Dlp &rhs) { return lhs * rhs; })
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
        return py::buffer_info(arg.data(), sizeof(double),
                               py::format_descriptor<double>::value(), 1,
                               {arg.Num}, {sizeof(double)});
      });
}

}  // namespace python

}  // namespace vsr
