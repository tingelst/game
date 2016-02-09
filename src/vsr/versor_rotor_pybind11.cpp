#include <pybind11/pybind11.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/cga_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddRotor(py::module &m) {
  py::class_<Rot>(m, "Rot")
      .def(py::init<double, double, double, double>())
      .def("__init__", [](Rot &instance,
                          Biv &arg) { new (&instance) Rot(Gen::rotor(arg)); },
           "Bivector logarithm: R = exp(B)")
      .def("log", [](const Rot &arg) { return Gen::log(arg); })
      .def("__mul__", [](const Rot &lhs, const Trs &rhs) { return lhs * rhs; })
      .def("__getitem__", &Rot::at);
}

} // namespace python

} // namespace vsr
