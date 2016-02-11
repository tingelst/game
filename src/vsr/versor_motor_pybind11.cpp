#include <sstream>
#include <pybind11/pybind11.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/cga_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddMotor(py::module &m) {
  py::class_<Mot>(m, "Mot")
      .def(py::init<double, double, double, double, double, double, double,
                    double>())
      .def("__getitem__", &Mot::at)
      .def("rev", &Mot::reverse)
      .def("inv", &Mot::inverse)
      .def("log", [](const Mot &arg) { return Gen::log(arg); })
      .def("__mul__", [](const Mot &lhs, double rhs) { return lhs * rhs; })
      .def("__mul__", [](const Mot &lhs, const Mot &rhs) { return lhs * rhs; })
      .def("__repr__",
           [](const Mot &arg) {
             std::stringstream ss;
             ss.precision(2);
             ss << "Mot: [";
             for (int i = 0; i < arg.Num; ++i) {
               ss << " " << arg[i];
             }
             ss << " ]";
             return ss.str();
           })
      .def_buffer([](Mot &arg) -> py::buffer_info {
        return py::buffer_info(arg.data(), sizeof(double),
                               py::format_descriptor<double>::value(), 1,
                               {arg.Num}, {sizeof(double)});
      });
}

}  // namespace python

}  // namespace vsr
