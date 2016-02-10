#include <pybind11/pybind11.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/cga_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddBivector(py::module &m) {
  py::class_<Biv>(m, "Biv")
      .def(py::init<double, double, double>())
      .def("__getitem__", &Biv::at)
      .def("duale", &Biv::duale)
      .def("unduale", &Biv::unduale)
      .def("unit", &Biv::unit)
      .def("rev", &Biv::reverse)
      .def("inv", &Biv::inverse)
      .def("spin", (Biv (Biv::*)(const Rot &) const) & Biv::spin)
      .def("exp", [](const Biv &biv) { return Gen::rotor(biv); })
      .def("__mul__", [](const Biv &lhs, double rhs) { return lhs * rhs; })
      .def("__repr__", [](const Biv &arg) {
        std::stringstream ss;
        ss.precision(4);
        ss << "Biv: [";
        for (int i = 0; i < arg.Num; ++i) {
          ss << " " << arg[i];
        }
        ss << " ]";
        return ss.str();
      });
}

}  // namespace python

}  // namespace vsr
