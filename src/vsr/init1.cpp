#include <pybind11/pybind11.h>
#include "game/vsr/cga_types.h"
#include "game/vsr/generic_op.h"

namespace py = pybind11;
using namespace vsr::cga;

void init1(const py::module& m) {
    py::class_<Vec>(m, "Vec")
       .def(py::init<double, double, double>())
       .def("spin", (Vec (Vec::*)(const Rot&) const) &Vec::spin)
       .def("null", &Vec::null);
}
