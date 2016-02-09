
#include <pybind11/pybind11.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/cga_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

PYBIND11_PLUGIN(libversor_cga_op) {
  py::module m("libversor_cga_op", "libversor_cga_op plugin");

  py::class_<Gen>(m, "Gen")
    .def_static("log", &Gen::logMotor)
    .def_static("mot", &Gen::mot)
    ;

    return m.ptr();
}

}  // namespace python

}  // namespace vsr
