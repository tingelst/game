#include <pybind11/pybind11.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/generic_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddPoint(py::module &m) {
  py::class_<Pnt>(m, "Pnt")
      .def(py::init<double, double, double, double, double>())
      .def("__getitem__", &Pnt::at)
      .def("spin", (Pnt (Pnt::*)(const Rot &) const) & Pnt::spin)
      .def("spin", (Pnt (Pnt::*)(const Mot &) const) & Pnt::spin)
      .def("__repr__",
           [](const Pnt &arg) {
             std::stringstream ss;
             ss.precision(4);
             ss << "Pnt: [";
             for (int i = 0; i < arg.Num; ++i) {
               ss << " " << arg[i];
             }
             ss << " ]";
             return ss.str();
           })
      .def_buffer([](Pnt &arg) -> py::buffer_info {
        return py::buffer_info(arg.data(), sizeof(double),
                               py::format_descriptor<double>::value(), 1,
                               {arg.Num}, {sizeof(double)});
      });
}

}  // namespace python

}  // namespace vsr
