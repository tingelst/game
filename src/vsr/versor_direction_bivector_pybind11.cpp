#include <pybind11/pybind11.h>

#include "game/vsr/cga_op.h"
#include "game/vsr/cga_types.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddDirectionBivector(py::module &m) {
  py::class_<Drb>(m, "Drb")
      .def(py::init<double, double, double>())
      .def("__getitem__", &Drb::at)
      .def("norm", &Drb::norm)
      .def("rnorm", &Drb::rnorm)
      .def("unit", &Drb::unit)
      .def("rev", &Drb::reverse)
      .def("inv", &Drb::inverse)
      .def("duale", &Drb::duale)
      .def("unduale", &Drb::unduale)
      .def("vec",
           [](const Drb &self) { return Vec(self[0], self[1], self[2]); })
      .def("spin", (Drb (Drb::*)(const Rot &) const) & Drb::spin)
      .def("spin", (Drb (Drb::*)(const Mot &) const) & Drb::spin)
      .def("__repr__",
           [](const Drb &arg) {
             std::stringstream ss;
             ss.precision(2);
             ss << "Drb: [";
             for (int i = 0; i < arg.Num; ++i) {
               ss << " " << arg[i];
             }
             ss << " ]";
             return ss.str();
           })
      .def_buffer([](Drb &arg) -> py::buffer_info {
        return py::buffer_info(
            arg.data(), sizeof(double), py::format_descriptor<double>::format(),
            1, {static_cast<unsigned long>(arg.Num)}, {sizeof(double)});
      });
}

} // namespace python

} // namespace vsr
