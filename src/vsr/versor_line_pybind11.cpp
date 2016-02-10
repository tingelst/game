#include <pybind11/pybind11.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/cga_op.h"

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
      .def("spin", (Lin (Lin::*)(const Rot &) const) & Lin::spin)
      .def("spin", (Lin (Lin::*)(const Mot &) const) & Lin::spin)
      .def("duale", &Lin::duale)
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
