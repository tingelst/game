#include <pybind11/pybind11.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/cga_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddVector(py::module &m) {

  py::class_<Vec>(m, "Vec")
      .def(py::init<double, double, double>())
      .def("__getitem__", &Vec::at)
      .def("unit", &Vec::unit)
      .def("rev", &Vec::reverse)
      .def("inv", &Vec::inverse)
      .def("trs", [](const Vec &arg) { return Gen::trs(arg); })
      .def("__mul__", [](const Vec &lhs, const Vec &rhs) { return lhs * rhs; })
      .def("spin", (Vec (Vec::*)(const Rot &) const) & Vec::spin)
      .def("null", &Vec::null)
      .def("__repr__", [](const Vec &arg) {
        std::stringstream ss;
        ss.precision(2);
        ss << "Vec: [";
        for (int i = 0; i < arg.Num; ++i) {
          ss << " " << arg[i];
        }
        ss << " ]";
        return ss.str();
      });
}

} // namespace python

} // namespace vsr
