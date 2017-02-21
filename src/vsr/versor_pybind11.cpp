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
void AddDirectionVector(py::module &m);
void AddDualPlane(py::module &m);
void AddPlane(py::module &m);
void AddGeneralRotor(py::module &m);
// void AddTangentVector(py::module &m);
void AddCircle(py::module &m);
void AddSphere(py::module &m);
void AddPointPair(py::module &m);
void AddTrivector(py::module &m);
void AddMultivectorDiff(py::module &m);
void AddCGA(py::module &m);
void AddEGA(py::module &m);
void AddReciprocalMotor(py::module &m);
  void AddE4(py::module &m);

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
  AddDirectionVector(m);
  AddDualPlane(m);
  AddPlane(m);
  AddGeneralRotor(m);
  // AddTangentVector(m);
  AddCircle(m);
  AddSphere(m);
  AddPointPair(m);
  AddTrivector(m);
  AddMultivectorDiff(m);
  AddCGA(m);
  AddEGA(m);
  AddReciprocalMotor(m);
  AddE4(m);

  return m.ptr();
}

}  // namespace python

}  // namespace vsr
