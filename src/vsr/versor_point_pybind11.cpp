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
      .def(
          "__init__",
          [](Pnt &instance) { new (&instance) Pnt(Vec(0.0, 0.0, 0.0).null()); })
      .def("__getitem__", &Pnt::at)
      .def("vec", [](const Pnt &self) { return Vec(self); })
      .def("__xor__", [](const Pnt &lhs, const Pnt &rhs) { return lhs ^ rhs; })
      .def("spin", (Pnt (Pnt::*)(const Rot &) const) & Pnt::spin)
      .def("spin", (Pnt (Pnt::*)(const Mot &) const) & Pnt::spin)
      .def("comm",
           [](const Pnt &lhs, const Dll &rhs) {
             return Dlp((lhs * rhs - rhs * lhs) * 0.5);
           })
      .def("acomm",
           [](const Pnt &lhs, const Dll &rhs) {
             return Dlp((lhs * rhs + rhs * lhs) * 0.5);
           })

      .def("comm",
           [](const Pnt &lhs, const Mot &rhs) {
             return Pnt((lhs * rhs - rhs * lhs) * 0.5);
           })
      .def("acomm",
           [](const Pnt &lhs, const Mot &rhs) {
             return Pnt((lhs * rhs + rhs * lhs) * 0.5);
           })
      .def("__repr__",
           [](const Pnt &arg) {
             std::stringstream ss;
             ss.precision(4);
             if (nga::Round::radius(arg) > 0.0) {
               ss << "Dls: [";
             } else {
               ss << "Pnt: [";
             }
             for (int i = 0; i < arg.Num; ++i) {
               ss << " " << arg[i];
             }
             ss << " ]";
             return ss.str();
           })
      .def_buffer([](Pnt &arg) -> py::buffer_info {
        return py::buffer_info(
            arg.data(), sizeof(double), py::format_descriptor<double>::value(),
            1, {static_cast<unsigned long>(arg.Num)}, {sizeof(double)});
      });
}

}  // namespace python

}  // namespace vsr
