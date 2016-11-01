#include <pybind11/pybind11.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/generic_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

using CGA = vsr::Multivector<
    vsr::algebra<vsr::metric<4, 1, true>, double>,
    vsr::Basis<(short)0, (short)1, (short)2, (short)4, (short)8, (short)16,
               (short)3, (short)5, (short)6, (short)9, (short)10, (short)12,
               (short)17, (short)18, (short)20, (short)24, (short)7, (short)11,
               (short)13, (short)14, (short)19, (short)21, (short)22, (short)25,
               (short)26, (short)28, (short)15, (short)23, (short)27, (short)29,
               (short)30, (short)31>>;

void AddPoint(py::module &m) {
  py::class_<Pnt>(m, "Pnt")
      .def(py::init<double, double, double, double, double>())
      .def(
          "__init__",
          [](Pnt &instance) { new (&instance) Pnt(Vec(0.0, 0.0, 0.0).null()); })
      .def("__init__",
           [](Pnt &instance, CGA &arg) { new (&instance) Pnt(arg); })
      .def("__getitem__", &Pnt::at)
      .def("dual", &Pnt::dual)
      .def("undual", &Pnt::undual)
      .def("vec", [](const Pnt &self) { return Vec(self); })
      .def("__xor__", [](const Pnt &lhs, const Pnt &rhs) { return lhs ^ rhs; })
      .def("spin", (Pnt (Pnt::*)(const Rot &) const) & Pnt::spin)
      .def("spin", (Pnt (Pnt::*)(const Trs &) const) & Pnt::spin)
      .def("spin", (Pnt (Pnt::*)(const Grt &) const) & Pnt::spin)
      .def("spin", (Pnt (Pnt::*)(const Mot &) const) & Pnt::spin)
      .def("__sub__",
           [](const Pnt &lhs, const Pnt &rhs) { return Dlp(lhs - rhs); })
      .def("__add__", [](const Pnt &lhs, const Pnt &rhs) { return lhs + rhs; })
      .def("__le__",
           [](const Pnt &lhs, const Pnt &rhs) { return (lhs <= rhs)[0]; })
      .def("comm",
           [](const Pnt &lhs, const Dll &rhs) {
             return Pnt((lhs * rhs - rhs * lhs) * 0.5);
           })
      .def("acomm",
           [](const Pnt &lhs, const Dll &rhs) {
             return Pnt((lhs * rhs + rhs * lhs) * 0.5);
           })

      .def("comm",
           [](const Pnt &lhs, const Mot &rhs) {
             return Pnt((lhs * rhs - rhs * lhs) * 0.5);
           })
      .def("acomm",
           [](const Pnt &lhs, const Mot &rhs) {
             return Pnt((lhs * rhs + rhs * lhs) * 0.5);
           })
      .def("reflect_in_line",
           [](const Pnt &self, const Dll &arg) {
             return Pnt(arg * self * !arg);
           })
      .def("reflect_in_plane",
           [](const Pnt &self, const Dlp &arg) {
             return Pnt(-arg * self * arg);
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
            arg.data(), sizeof(double), py::format_descriptor<double>::format(),
            1, {static_cast<unsigned long>(arg.Num)}, {sizeof(double)});
      });
}

} // namespace python

} // namespace vsr
