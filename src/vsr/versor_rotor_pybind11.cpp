#include <pybind11/pybind11.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/cga_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddRotor(py::module &m) {
  py::class_<Rot>(m, "Rot")
      .def(py::init<double, double, double, double>())
      .def("__init__", [](Rot &instance,
                          Biv &arg) { new (&instance) Rot(Gen::rotor(arg)); },
           "Bivector logarithm: R = exp(B)")
      .def("log", [](const Rot &arg) { return Gen::log(arg); })
      .def("__mul__", [](const Rot &lhs, const Trs &rhs) { return lhs * rhs; })
      .def("__mul__", [](const Rot &lhs, const Rot &rhs) { return lhs * rhs; })
      .def("__getitem__", &Rot::at)
      .def("__repr__",
           [](const Rot &arg) {
             std::stringstream ss;
             ss.precision(2);
             ss << "Rot: [";
             for (int i = 0; i < arg.Num; ++i) {
               ss << " " << arg[i];
             }
             ss << " ]";
             return ss.str();
           })
      .def_buffer([](Rot &arg) -> py::buffer_info {
        return py::buffer_info(arg.data(), sizeof(double),
                               py::format_descriptor<double>::value(), 1,
                               {static_cast<unsigned long>(arg.Num)}, {sizeof(double)});
      });
}

}  // namespace python

}  // namespace vsr
