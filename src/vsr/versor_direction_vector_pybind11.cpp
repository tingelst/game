#include <pybind11/pybind11.h>

#include "game/vsr/cga_op.h"
#include "game/vsr/cga_types.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddDirectionVector(py::module &m) {
  py::class_<Drv>(m, "Drv")
      .def(py::init<double, double, double>())
      .def("__getitem__", &Drv::at)
      .def("__setitem__", [](Drv &arg, int idx, double val) { arg[idx] = val; })
      .def("norm", &Drv::norm)
      .def("rnorm", &Drv::rnorm)
      .def("unit", &Drv::unit)
      .def("rev", &Drv::reverse)
      .def("inv", &Drv::inverse)
      .def("duale", &Drv::duale)
      .def("unduale", &Drv::unduale)
      .def("vec",
           [](const Drv &self) { return Vec(self[0], self[1], self[2]); })
      .def("spin", (Drv (Drv::*)(const Rot &) const) & Drv::spin)
      .def("spin", (Drv (Drv::*)(const Mot &) const) & Drv::spin)
      .def("comm", [](const Drv &lhs,
                      const Pnt &rhs) { return Pnt(lhs * rhs - rhs * lhs) * 0.5; })
      .def("__repr__",
           [](const Drv &arg) {
             std::stringstream ss;
             ss.precision(2);
             ss << "Drv: [";
             for (int i = 0; i < arg.Num; ++i) {
               ss << " " << arg[i];
             }
             ss << " ]";
             return ss.str();
           })
      .def_buffer([](Drv &arg) -> py::buffer_info {
        return py::buffer_info(
            arg.data(), sizeof(double), py::format_descriptor<double>::format(),
            1, {static_cast<unsigned long>(arg.Num)}, {sizeof(double)});
      });
}

} // namespace python

} // namespace vsr
