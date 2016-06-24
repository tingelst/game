#include <pybind11/pybind11.h>

#include "game/vsr/cga_op.h"
#include "game/vsr/cga_types.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddLine(py::module &m) {
  py::class_<Lin>(m, "Lin")
      .def(py::init<double, double, double, double, double, double>())
      .def("__init__",
           [](Lin &instance, const Vec &arg1, const Vec &arg2) {
             new (&instance) Lin(Construct::line(arg1, arg2));
           })
      .def("__init__",
           [](Lin &instance, const Pnt &arg1, const Pnt &arg2) {
             new (&instance) Lin(Construct::line(arg1, arg2));
           })
      .def("__init__",
           [](Lin &instance, const Pnt &arg1, const Vec &arg2) {
             new (&instance) Lin(arg1 ^ arg2 ^ Inf(1.0));
           })
      .def("dir", [](const Lin &arg) { return -(Ori(1.0) ^ Inf(1.0)) <= arg; })
    .def("biv", [](const Lin &arg) { return -(Ori(1.0) ^ Inf(1.0)) <= (arg ^ Ori(1.)); })
      .def("spin", (Lin (Lin::*)(const Rot &) const) & Lin::spin)
      .def("spin", (Lin (Lin::*)(const Mot &) const) & Lin::spin)
      .def("duale", &Lin::duale)
      .def("__mul__", [](const Lin &lhs, const Lin &rhs) { return lhs * rhs; })
      .def("unduale", &Lin::unduale)
      .def("dual", &Lin::dual)
      .def("undual", &Lin::undual)
      .def("__repr__", [](const Lin &arg) {
        std::stringstream ss;
        ss.precision(4);
        ss << "Lin: [";
        for (int i = 0; i < arg.Num; ++i) {
          ss << " " << arg[i];
        }
        ss << " ]";
        return ss.str();
      });
}

} // namespace python

} // namespace vsr
