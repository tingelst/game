#include <pybind11/pybind11.h>
#include <sstream>

#include "game/vsr/cga_op.h"
#include "game/vsr/cga_types.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

using CGA = vsr::Multivector<
    vsr::algebra<vsr::metric<4, 1, true>, double>,
    vsr::Basis<(short)0, (short)1, (short)2, (short)4, (short)8, (short)16,
               (short)3, (short)5, (short)6, (short)9, (short)10, (short)12,
               (short)17, (short)18, (short)20, (short)24, (short)7, (short)11,
               (short)13, (short)14, (short)19, (short)21, (short)22, (short)25,
               (short)26, (short)28, (short)15, (short)23, (short)27, (short)29,
               (short)30, (short)31>>;

using MotRec = vsr::Multivector<vsr::algebra<vsr::metric<4, 1, true>, double>,
                                vsr::Basis<0, 3, 5, 6, 9, 10, 12, 15>>;
void AddReciprocalMotor(py::module &m) {
  py::class_<MotRec>(m, "MotRec")
      .def(py::init<double, double, double, double, double, double, double,
                    double>())
      .def("__init__",
           [](MotRec &instance, CGA &arg) { new (&instance) MotRec(arg); })
      .def("__getitem__", &MotRec::at)
      .def("__setitem__",
           [](MotRec &arg, int idx, double val) { arg[idx] = val; })
      .def("rev", &MotRec::reverse)
      .def("inv", &MotRec::inverse)
      .def("comm", [](const MotRec &lhs,
                      const Dll &rhs) { return (lhs * rhs - rhs * lhs) * 0.5; })
      .def("acomm",
           [](const MotRec &lhs, const Dll &rhs) {
             return (lhs * rhs + rhs * lhs) * 0.5;
           })
      .def("comm",
           [](const MotRec &lhs, const MotRec &rhs) {
             return (lhs * rhs - rhs * lhs) * 0.5;
           })
      .def("acomm",
           [](const MotRec &lhs, const MotRec &rhs) {
             return (lhs * rhs + rhs * lhs) * 0.5;
           })
      .def("__add__",
           [](const MotRec &lhs, const MotRec &rhs) { return lhs + rhs; })
      .def("__sub__",
           [](const MotRec &lhs, const MotRec &rhs) { return lhs - rhs; })
      .def("__mul__", [](const MotRec &lhs, double rhs) { return lhs * rhs; })
      .def("__mul__",
           [](const MotRec &lhs, const MotRec &rhs) { return lhs * rhs; })
      .def("__mul__",
           [](const MotRec &lhs, const Mot &rhs) { return lhs * rhs; })
      .def("__mul__",
           [](const MotRec &lhs, const Dll &rhs) { return lhs * rhs; })
      .def("__repr__",
           [](const MotRec &arg) {
             std::stringstream ss;
             ss.precision(2);
             ss << "MotRec: [";
             for (int i = 0; i < arg.Num; ++i) {
               ss << " " << arg[i];
             }
             ss << " ]";
             return ss.str();
           })
      .def_buffer([](MotRec &arg) -> py::buffer_info {
        return py::buffer_info(
            arg.data(), sizeof(double), py::format_descriptor<double>::format(),
            1, {static_cast<unsigned long>(arg.Num)}, {sizeof(double)});
      });
}

} // namespace python

} // namespace vsr
