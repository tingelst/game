#include <pybind11/pybind11.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/generic_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddOrigin(py::module &m) {
  py::class_<Ori>(m, "Ori")
      .def(py::init<double>());
}

} // namespace python

} // namespace vsr
