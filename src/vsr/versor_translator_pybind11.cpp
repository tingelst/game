#include <pybind11/pybind11.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/generic_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddTranslator(py::module &m) {
  py::class_<Trs>(m, "Trs")
      .def(py::init<double, double, double, double>())
      .def("__getitem__", &Trs::at)
      .def("__setitem__", [](Trs &arg, int idx, double val) { arg[idx] = val; })
      .def("rev", &Trs::reverse)
      .def("inv", &Trs::inverse)
      .def("__mul__", [](const Trs &lhs, const Rot &rhs) { return lhs * rhs; })
      .def("__repr__", [](const Trs &arg) {
        std::stringstream ss;
        ss.precision(4);
        ss << "Trs: [";
        for (int i = 0; i < arg.Num; ++i) {
          ss << " " << arg[i];
        }
        ss << " ]";
        return ss.str();
      });
}

}  // namespace python

}  // namespace vsr
