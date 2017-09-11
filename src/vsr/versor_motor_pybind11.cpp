#include <pybind11/pybind11.h>
#include <sstream>

#include "game/vsr/cga_op.h"
#include "game/vsr/cga_types.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

template <typename T> DualLine<T> log(const Motor<T> &m) {
  DualLine<T> q(m);
  T ac = acos(m[0]);
  T den = sin(ac) / ac;
  T den2 = ac * ac * den;

  if (den2 > T(0.0)) {
    DualLine<T> b = Bivector<T>(m) / den;
    DualLine<T> c_perp = -b * DirectionTrivector<T>(m) / den2;
    DualLine<T> c_para = -b * DualLine<T>(b * q) / den2;
    return b + c_perp + c_para;
  } else {
    return q;
  }
}

using CGA = vsr::Multivector<
    vsr::algebra<vsr::metric<4, 1, true>, double>,
    vsr::Basis<(short)0, (short)1, (short)2, (short)4, (short)8, (short)16,
               (short)3, (short)5, (short)6, (short)9, (short)10, (short)12,
               (short)17, (short)18, (short)20, (short)24, (short)7, (short)11,
               (short)13, (short)14, (short)19, (short)21, (short)22, (short)25,
               (short)26, (short)28, (short)15, (short)23, (short)27, (short)29,
               (short)30, (short)31>>;

void AddMotor(py::module &m) {
  py::class_<Mot>(m, "Mot")
      .def(py::init<double, double, double, double, double, double, double,
                    double>())
      .def("__init__",
           [](Mot &instance, CGA &arg) { new (&instance) Mot(arg); })
      .def("__getitem__", &Mot::at)
      .def("__setitem__", [](Mot &arg, int idx, double val) { arg[idx] = val; })
      .def("rev", &Mot::reverse)
      .def("inv", &Mot::inverse)
      .def("dll", [](const Mot &arg) { return Dll(arg); })
      .def("retract",
           [](const Mot &arg) {
             double norm = arg.norm();
             Mot b = arg * ~arg;
             double s0 = b[0];
             double s4 = b[7];
             auto Sinv = Scalar<double>{(1.0) / norm} *
                         (Scalar<double>{(1.0)} +
                          DirectionTrivector<double>{-(s4 / (2.0 * s0))});
             return arg * Sinv;
           })
      .def("log", [](const Mot &arg) { return Gen::log(arg); })
      .def("log2", [](const Mot &arg) { return log<double>(arg); })
      .def("comm", [](const Mot &lhs,
                      const Dll &rhs) { return (lhs * rhs - rhs * lhs) * 0.5; })
      .def("comm", [](const Mot &lhs,
                      const Pnt &rhs) { return (lhs * rhs - rhs * lhs) * 0.5; })
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
      .def("__add__",
           [](const Mot &lhs, const double &rhs) { return lhs + rhs; })
      .def("__radd__",
           [](const Mot &lhs, const double &rhs) { return lhs + rhs; })
      .def("__add__", [](const Mot &lhs, const Dll &rhs) { return lhs + rhs; })
      .def("__radd__", [](const Mot &lhs, const Dll &rhs) { return lhs + rhs; })
      .def("__sub__", [](const Mot &lhs, const Mot &rhs) { return lhs - rhs; })
      .def("__mul__", [](const Mot &lhs, double rhs) { return lhs * rhs; })
      .def("__mul__", [](const Mot &lhs, const Mot &rhs) { return lhs * rhs; })
      .def("__mul__", [](const Mot &lhs, const Dll &rhs) { return lhs * rhs; })
      .def("__mul__", [](const Mot &lhs, const Trs &rhs) { return lhs * rhs; })
      .def("__xor__",
           [](const Mot &lhs, const Dll &rhs) { return Mot(lhs ^ rhs); })
      .def("__le__",
           [](const Mot &lhs, const Mot &rhs) { return (lhs <= rhs)[0]; })
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
            arg.data(), sizeof(double), py::format_descriptor<double>::format(),
            1, {static_cast<unsigned long>(arg.Num)}, {sizeof(double)});
      });
}

} // namespace python

} // namespace vsr
