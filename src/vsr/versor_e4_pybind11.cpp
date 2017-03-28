#include <pybind11/pybind11.h>
#include <sstream>

#include "game/vsr/cga_op.h"
#include "game/vsr/cga_types.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

using E4 = vsr::Multivector<vsr::algebra<vsr::metric<4, 0, false>, double>,
                            vsr::Basis<0, 1, 2, 4, 8, 3, 5, 6, 9, 10, 12, 7, 11, 13, 14, 15>>;

void AddE4(py::module &m) {
  py::class_<E4>(m, "E4")
    .def(py::init<double, double, double, double, double, double, double, double,
                    double, double, double, double, double, double, double, double>())
      .def("__init__",
           [](E4 &instance, E4 &arg) { new (&instance) E4(arg); })
      .def("__getitem__", &E4::at)
      .def("__setitem__", [](E4 &arg, int idx, double val) { arg[idx] = val; })
      .def("rev", &E4::reverse)
      .def("inv", &E4::inverse)
      .def("norm", &E4::norm)
      .def("rnorm", &E4::rnorm)
      .def("unit", &E4::unit)
      .def("dual", &E4::duale)
      .def("undual", &E4::unduale)
      .def("comm",
           [](const E4 &lhs, const E4 &rhs) {
             return E4(lhs * rhs - rhs * lhs) * 0.5;
           })
      .def("acomm",
           [](const E4 &lhs, const E4 &rhs) {
             return E4(lhs * rhs + rhs * lhs) * 0.5;
           })
      .def("__add__",
           [](const E4 &lhs, const E4 &rhs) { return E4(lhs + rhs); })
      .def("__sub__",
           [](const E4 &lhs, const E4 &rhs) { return E4(lhs - rhs); })
      .def("__mul__",
           [](const E4 &lhs, const E4 &rhs) { return E4(lhs * rhs); })
      .def("__mul__",
           [](const E4 &lhs, double &rhs) { return E4(lhs * rhs); })
      .def("__rmul__",
           [](const double &lhs, E4 &rhs) { return E4(rhs * lhs); })
      .def("__le__",
           [](const E4 &lhs, const E4 &rhs) { return E4(lhs <= rhs); })
      .def("__xor__",
           [](const E4 &lhs, const E4 &rhs) { return E4(lhs ^ rhs); })
    .def("comm",
         [](const E4 &lhs, const E4 &rhs) { return E4(lhs % rhs); })
    .def("acomm",
         [](const E4 &lhs, const E4 &rhs) { return E4(lhs * rhs - rhs * lhs) * 0.5; })
      .def("spin", (E4 (E4::*)(const E4 &) const) & E4::spin)
      .def("__repr__",
           [](const E4 &arg) {
             std::stringstream ss;
             ss.precision(2);
             ss << "E4: [";
             for (int i = 0; i < arg.Num; ++i) {
               ss << " " << arg[i];
             }
             ss << " ]";
             return ss.str();
           })
      .def_buffer([](E4 &arg) -> py::buffer_info {
        return py::buffer_info(
            arg.data(), sizeof(double), py::format_descriptor<double>::format(),
            1, {static_cast<unsigned long>(arg.Num)}, {sizeof(double)});
      });
}

} // namespace python

} // namespace vsr
