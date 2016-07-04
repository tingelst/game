#include <pybind11/pybind11.h>
#include <sstream>

#include "game/vsr/cga_op.h"
#include "game/vsr/cga_types.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddMotor(py::module &m) {
  py::class_<Mot>(m, "Mot")
      .def(py::init<double, double, double, double, double, double, double,
                    double>())
      .def("__getitem__", &Mot::at)
      .def("__setitem__", [](Mot &arg, int idx, double val) { arg[idx] = val; })
      .def("rev", &Mot::reverse)
      .def("inv", &Mot::inverse)
      .def("log", [](const Mot &arg) { return Gen::log(arg); })
      .def("comm", [](const Mot &lhs,
                      const Dll &rhs) { return (lhs * rhs - rhs * lhs) * 0.5; })
      .def("acomm",
           [](const Mot &lhs, const Dll &rhs) {
             return (lhs * rhs + rhs * lhs) * 0.5;
           })
      .def("comm", [](const Mot &lhs,
                      const Mot &rhs) { return (lhs * rhs - rhs * lhs) * 0.5; })
      .def("acomm",
           [](const Mot &lhs, const Mot &rhs) {
             return (lhs * rhs + rhs * lhs) * 0.5;
           })
      .def("rot",
           [](const Mot &self) {
             Rot R{self[0], self[1], self[2], self[3]};
             return R;
           })
      .def("trs",
           [](const Mot &self) {
             Rot R{self[0], self[1], self[2], self[3]};
             Vec t = ((Ori(1.0) <= self) / R) * -2.0;
             return t;
           })
      .def("ratio", [](const Mot &lhs, const Mot &rhs,
                       double t) { return Mot(Gen::log(lhs, rhs, t)); })
      .def("__add__", [](const Mot &lhs, const Mot &rhs) { return lhs + rhs; })
      .def("__sub__", [](const Mot &lhs, const Mot &rhs) { return lhs - rhs; })
      .def("__mul__", [](const Mot &lhs, double rhs) { return lhs * rhs; })
      .def("__mul__", [](const Mot &lhs, const Mot &rhs) { return lhs * rhs; })
      .def("__mul__", [](const Mot &lhs, const Dll &rhs) { return lhs * rhs; })
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
        return py::buffer_info(
            arg.data(), sizeof(double), py::format_descriptor<double>::value(),
            1, {static_cast<unsigned long>(arg.Num)}, {sizeof(double)});
      });
}

}  // namespace python

}  // namespace vsr
