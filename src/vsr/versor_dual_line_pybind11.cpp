#include <pybind11/pybind11.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/cga_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddDualLine(py::module &m) {
  py::class_<Dll>(m, "Dll")
      .def(py::init<double, double, double, double, double, double>())
      .def("__init__",
           [](Dll &instance, const Vec &arg1, const Vec &arg2) {
             new (&instance) Dll(Construct::line(arg1, arg2).dual());
           })

      .def("__getitem__", &Dll::at)
      .def("__setitem__", [](Dll &arg, int idx, double val) { arg[idx] = val; })
      .def("comm", [](const Dll &lhs,
                      const Mot &rhs) { return (lhs * rhs - rhs * lhs) * 0.5; })
      .def("acomm",
           [](const Dll &lhs, const Mot &rhs) {
             return (lhs * rhs + rhs * lhs) * 0.5;
           })
      .def("comm", [](const Dll &lhs,
                      const Pnt &rhs) { return Pnt((lhs * rhs - rhs * lhs) * 0.5); })
      .def("acomm",
           [](const Dll &lhs, const Pnt &rhs) {
             return Pnt((lhs * rhs + rhs * lhs) * 0.5);
           })
      .def("duale", &Dll::duale)
      .def("unduale", &Dll::unduale)
      .def("dual", &Dll::dual)
      .def("undual", &Dll::undual)
      .def("unit", &Dll::unit)
      .def("rev", &Dll::reverse)
      .def("inv", &Dll::inverse)
      .def("spin", (Dll (Dll::*)(const Mot &) const) & Dll::spin)
      .def("exp", [](const Dll &arg) { return Gen::motor(arg); })
      .def("__neg__", [](const Dll &arg) { return -arg; })
      .def("__mul__", [](const Dll &lhs, const Mot &rhs) { return lhs * rhs; })
      .def("__mul__", [](const Dll &lhs, double rhs) { return lhs * rhs; })
      .def("__div__", [](const Dll &lhs, double rhs) { return lhs / rhs; })
      .def("__repr__",
           [](const Dll &arg) {
             std::stringstream ss;
             ss.precision(4);
             ss << "Dll: [";
             for (int i = 0; i < arg.Num; ++i) {
               ss << " " << arg[i];
             }
             ss << " ]";
             return ss.str();
           })
      .def_buffer([](Dll &arg) -> py::buffer_info {
        return py::buffer_info(
            arg.data(), sizeof(double), py::format_descriptor<double>::value(),
            1, {static_cast<unsigned long>(arg.Num)}, {sizeof(double)});
      });
}

}  // namespace python

}  // namespace vsr
