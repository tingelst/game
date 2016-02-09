#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/generic_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

// Forward declarations
void AddScalar(py::module &m);
void AddVector(py::module &m);
void AddBivector(py::module &m);
void AddRotor(py::module &m);
void AddPoint(py::module &m);
void AddLine(py::module &m);
void AddDualLine(py::module &m);
void AddTranslator(py::module &m);
void AddMotor(py::module &m);
void AddOrigin(py::module &m);
void AddInfinity(py::module &m);

PYBIND11_PLUGIN(versor_pybind11) {

  py::module m("versor_pybind11", "versor plugin");
  AddVector(m);
  AddBivector(m);
  AddRotor(m);
  AddPoint(m);
  AddLine(m);
  AddDualLine(m);
  AddTranslator(m);
  AddMotor(m);
  AddOrigin(m);
  AddInfinity(m);

  return m.ptr();
}

} // namespace python

} // namespace vsr
