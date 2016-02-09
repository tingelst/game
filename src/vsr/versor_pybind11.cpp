#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/generic_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;


void AddRotor(py::module& m)
{
  py::class_<Rot>(m, "Rot")
    .def(py::init<double, double, double, double>())
    ;
}

void AddVector(py::module& m)
{

  py::class_<Vec>(m, "Vec")
    .def(py::init<double, double, double>())
    .def("__getitem__", &Vec::at)                                \
    .def("__mul__", [](const Vec& lhs, const Vec& rhs){ return lhs * rhs;})
    .def("spin", (Vec (Vec::*)(const Rot&) const) &Vec::spin)
    .def("null", &Vec::null)
    ;


}

PYBIND11_PLUGIN(versor_pybind11) {

  py::module m("versor_pybind11", "versor plugin");
  AddVector(m);
  AddRotor(m);

  return m.ptr();

}



}  // namespace python

}  // namespace vsr
