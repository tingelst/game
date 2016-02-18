#include <pybind11/pybind11.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/cga_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddCircle(py::module &m) {
  py::class_<Cir>(m, "Cir")
      .def(py::init<double, double, double, double, double, double, double,
                    double, double, double>())
      .def("__init__",
           [](Cir &instance, const Pnt &p, double radius, const Biv &biv) {
             new (&instance) Cir(Construct::circle(p, radius, biv));
           })
      .def("duale", &Cir::duale)
      .def("unduale", &Cir::unduale)
      .def("dual", &Cir::dual)
      .def("undual", &Cir::undual)
      .def("unit", &Cir::unit)
      .def("rev", &Cir::reverse)
      .def("inv", &Cir::inverse)
      .def("pnt", [](const Cir &self) { return Round::location(self); })
      .def("radius", [](const Cir &self) { return Round::radius(self); })
      .def("pln", [](const Cir &self) { return Round::carrier(self); })
      .def("spin", (Cir (Cir::*)(const Mot &) const) & Cir::spin)
      .def("__mul__", [](const Cir &lhs, double rhs) { return lhs * rhs; })
      .def("__div__", [](const Cir &lhs, double rhs) { return lhs / rhs; })
      .def("__repr__",
           [](const Cir &arg) {
             std::stringstream ss;
             ss.precision(4);
             ss << "Cir: [";
             for (int i = 0; i < arg.Num; ++i) {
               ss << " " << arg[i];
             }
             ss << " ]";
             return ss.str();
           })
      .def_buffer([](Cir &arg) -> py::buffer_info {
        return py::buffer_info(
            arg.data(), sizeof(double), py::format_descriptor<double>::value(),
            1, {static_cast<unsigned long>(arg.Num)}, {sizeof(double)});
      });
}

}  // namespace python

}  // namespace vsr
