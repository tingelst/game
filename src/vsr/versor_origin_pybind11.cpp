#include <pybind11/pybind11.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/generic_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddOrigin(py::module& m) {
  py::class_<Ori>(m, "Ori")
      .def(py::init<double>())
      .def("op",
           [](const Ori& self, const Vec& other) { return self ^ other; });
}

}  // namespace python

}  // namespace vsr
