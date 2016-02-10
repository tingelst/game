#include <pybind11/pybind11.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/cga_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddDualLine(py::module &m) {
  py::class_<Dll>(m, "Dll")
      .def(py::init<double, double, double, double, double, double>())
      .def("__init__",
           [](Dll &instance, const Vec &arg1, const Vec &arg2) {
             new (&instance) Dll(Construct::line(arg1, arg2).dual());
           })
      .def("duale", &Dll::duale)
      .def("unduale", &Dll::unduale)
      .def("dual", &Dll::dual)
      .def("undual", &Dll::undual)
      .def("unit", &Dll::unit)
      .def("rev", &Dll::reverse)
      .def("inv", &Dll::inverse)
      .def("spin", (Dll (Dll::*)(const Mot &) const) & Dll::spin)
      .def("exp", [](const Dll &arg) { return Gen::motor(arg); })
      .def("__repr__", [](const Dll &arg) {
        std::stringstream ss;
        ss.precision(4);
        ss << "Dll: [";
        for (int i = 0; i < arg.Num; ++i) {
          ss << " " << arg[i];
        }
        ss << " ]";
        return ss.str();
      });
}

}  // namespace python

}  // namespace vsr
