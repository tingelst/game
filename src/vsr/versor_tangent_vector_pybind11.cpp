#include <pybind11/pybind11.h>

#include "game/vsr/cga_op.h"
#include "game/vsr/cga_types.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddTangentVector(py::module &m) {
  py::class_<Tnv>(m, "Tnv")
      .def(py::init<>())
      // .def("__init__",
      //      [](Tnv &instance, const Pnt &pnt, const Vec &vec) {
      //        new (&instance) Tnv(pnt ^ (-pnt <= (vec * Inf(1.0))));
      //      })
      // .def("__init__",
      //      [](Tnv &instance, const Vec &arg) {
      //        new (&instance) Tnv(Ori(1.0) ^ arg);
      //      })
      // .def("__getitem__", &Tnv::at)
      // .def("norm", &Tnv::norm)
      // .def("rnorm", &Tnv::rnorm)
      // .def("unit", &Tnv::unit)
      // .def("rev", &Tnv::reverse)
      // .def("inv", &Tnv::inverse)
      // .def("dual", &Tnv::dual)
      // .def("undual", &Tnv::undual)
      // .def("__sub__", [](const Tnv &lhs, const Tnv &rhs) { return lhs - rhs;
      // })
      // .def("pnt", [](const Tnv &self) { return Pnt(self / (-Inf(1) <= self));
      // })
      // .def("drv", [](const Tnv &self) { return -(Inf(1) <= self) ^ Inf(1); })
      // .def("spin", (Tnv (Tnv::*)(const Rot &) const) & Tnv::spin)
      // .def("spin", (Tnv (Tnv::*)(const Mot &) const) & Tnv::spin)
      .def("spin",
           [](const Tnv &self, const Mot &mot) { return (Pnt() ^ (-Pnt() <= (Vec() * Inf()))).print() ; })
      // .def("lin", [](const Tnv &self) { return self ^ Inf(1); })
      // .def("__mul__", [](const Tnv &lhs, double rhs) { return lhs * rhs; })
      // .def("__mul__", [](const Tnv &lhs, const Tnv &rhs) { return lhs * rhs;
      // })
      .def("__repr__",
           [](const Tnv &arg) {
             std::stringstream ss;
             ss.precision(2);
             ss << "Tnv: [";
             for (int i = 0; i < arg.Num; ++i) {
               ss << " " << arg[i];
             }
             ss << " ]";
             return ss.str();
           })
      .def_buffer([](Tnv &arg) -> py::buffer_info {
        return py::buffer_info(
            arg.data(), sizeof(double), py::format_descriptor<double>::format(),
            1, {static_cast<unsigned long>(arg.Num)}, {sizeof(double)});
      });
}

} // namespace python

} // namespace vsr
