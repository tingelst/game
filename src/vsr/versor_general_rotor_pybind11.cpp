#include <sstream>
#include <pybind11/pybind11.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/cga_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddGeneralRotor(py::module &m) {
  py::class_<Grt>(m, "Grt")
      .def(py::init<double, double, double, double, double, double, double>())
      .def("__getitem__", &Grt::at)
      .def("rev", &Grt::reverse)
      .def("inv", &Grt::inverse)
      .def("__mul__", [](const Grt &lhs, double rhs) { return lhs * rhs; })
      .def("__mul__", [](const Grt &lhs, const Dlp &rhs) { return lhs * rhs; })
      .def("__mul__", [](const Grt &lhs, const Grt &rhs) { return lhs * rhs; })
      .def("__repr__",
           [](const Grt &arg) {
             std::stringstream ss;
             ss.precision(2);
             ss << "Grt: [";
             for (int i = 0; i < arg.Num; ++i) {
               ss << " " << arg[i];
             }
             ss << " ]";
             return ss.str();
           })
      .def_buffer([](Grt &arg) -> py::buffer_info {
        return py::buffer_info(
            arg.data(), sizeof(double), py::format_descriptor<double>::format(),
            1, {static_cast<unsigned long>(arg.Num)}, {sizeof(double)});
      });
}

}  // namespace python

}  // namespace vsr
