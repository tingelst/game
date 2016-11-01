#include <pybind11/pybind11.h>

#include "game/vsr/cga_op.h"
#include "game/vsr/cga_types.h"
#include "game/vsr/generic_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddSphere(py::module &m) {
  py::class_<Sph>(m, "Sph")
      .def(py::init<double, double, double, double, double>())
      .def("__getitem__", &Sph::at)
      .def("__init__",
           [](Sph &instance, const Pnt &p, double radius) {
             new (&instance) Sph(nga::Round::sphere(p, radius).undual());
           })
      .def("__init__",
           [](Sph &instance, const Pnt &p1, const Pnt &p2, const Pnt &p3,
              const Pnt &p4) { new (&instance) Sph(p1 ^ p2 ^ p3 ^ p4); })
      .def("duale", &Sph::duale)
      .def("unduale", &Sph::unduale)
      .def("dual", &Sph::dual)
      .def("undual", &Sph::undual)
      .def("unit", &Sph::unit)
      .def("rev", &Sph::reverse)
      .def("inv", &Sph::inverse)
      .def("spin", (Sph (Sph::*)(const Rot &) const) & Sph::spin)
      .def("spin", (Sph (Sph::*)(const Mot &) const) & Sph::spin)
      .def("pnt", [](const Sph &self) { return nga::Round::location(self); })
      .def("radius", [](const Sph &self) { return nga::Round::radius(self); })
      .def("meet",
           [](const Sph &self, const Sph &other) {
             return Construct::meet(self.dual(), other.dual());
           })
      .def("__repr__",
           [](const Sph &arg) {
             std::stringstream ss;
             ss.precision(4);
             ss << "Sph: [";
             for (int i = 0; i < arg.Num; ++i) {
               ss << " " << arg[i];
             }
             ss << " ]";
             return ss.str();
           })
      .def_buffer([](Sph &arg) -> py::buffer_info {
        return py::buffer_info(
            arg.data(), sizeof(double), py::format_descriptor<double>::format(),
            1, {static_cast<unsigned long>(arg.Num)}, {sizeof(double)});
      });
}

} // namespace python

} // namespace vsr
