#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

#include "game/vsr/cga_types.h"
#include "game/vsr/generic_op.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

using ScaDrt = decltype(Sca{1.0} + Drt{1.0});
using TrfTnv = decltype(Tnv{1.0, 1.0, 1.0}.spin(Mot()));
using VecTri = decltype(Vec() + Tri());

#define GAME_VSR_WRAP_MULTIVECTOR(MULTIVECTOR, ...)                                                            \
  MULTIVECTOR (MULTIVECTOR::*MULTIVECTOR##SpinByRotor)(const Rot&) const = \
      &MULTIVECTOR::spin;         \
  MULTIVECTOR (MULTIVECTOR::*MULTIVECTOR##SpinByMotor)(const Mot&) const = \
      &MULTIVECTOR::spin;         \
  MULTIVECTOR (MULTIVECTOR::*MULTIVECTOR##SpinByGeneralRotor)(const Grt&)  \
      const = &MULTIVECTOR::spin; \
  py::class_<MULTIVECTOR>(m, #MULTIVECTOR)                                                                     \
      .def(py::init<MULTIVECTOR>())                                                                            \
      .def("__getitem__", &MULTIVECTOR::at)                                                                    \
      .def("unit", &MULTIVECTOR::unit)                                                                         \
      .def("runit", &MULTIVECTOR::runit)                                                                       \
      .def("tunit", &MULTIVECTOR::tunit)                                                                       \
      .def("wt", &MULTIVECTOR::wt)                                                                             \
      .def("rwt", &MULTIVECTOR::rwt)                                                                           \
      .def("norm", &MULTIVECTOR::norm)                                                                         \
      .def("rnorm", &MULTIVECTOR::rnorm)                                                                       \
      .def("rev", &MULTIVECTOR::reverse)                                                                       \
      .def("inv", &MULTIVECTOR::inverse)                                                                       \
      .def("duale", &MULTIVECTOR::duale)                                                                       \
      .def("unduale", &MULTIVECTOR::unduale)                                                                   \
      .def("dual", &MULTIVECTOR::dual)                                                                         \
      .def("undual", &MULTIVECTOR::undual)                                                                     \
      .def("spin", MULTIVECTOR##SpinByRotor)                                                                   \
      .def("spin", MULTIVECTOR##SpinByMotor)                                                                   \
      .def("spin", MULTIVECTOR##SpinByGeneralRotor)                                                            \
      .def("__neg__", &MULTIVECTOR::negate)                                                                    \
      .def(py::self == py::self)                                                                               \
      .def(py::self *py::other<double>())                                                                      \
      .def(py::self / py::other<double>())                                                                     \
      .def(py::self + py::other<double>())                                                                     \
      .def(py::self <= py::self)                                                                               \
      .def(py::self <= py::other<Sca>())                                                                       \
      .def(py::self <= py::other<Vec>())                                                                       \
      .def(py::self <= py::other<Biv>())                                                                       \
      .def(py::self <= py::other<Rot>())                                                                       \
      .def(py::self <= py::other<Tri>())                                                                       \
      .def(py::self <= py::other<Ori>())                                                                       \
      .def(py::self <= py::other<Inf>())                                                                       \
      .def(py::self <= py::other<Mnk>())                                                                       \
      .def(py::self <= py::other<Pss>())                                                                       \
      .def(py::self <= py::other<Pnt>())                                                                       \
      .def(py::self <= py::other<Par>())                                                                       \
      .def(py::self <= py::other<Cir>())                                                                       \
      .def(py::self <= py::other<Sph>())                                                                       \
      .def(py::self <= py::other<Dls>())                                                                       \
      .def(py::self <= py::other<Flp>())                                                                       \
      .def(py::self <= py::other<Dll>())                                                                       \
      .def(py::self <= py::other<Lin>())                                                                       \
      .def(py::self <= py::other<Dlp>())                                                                       \
      .def(py::self <= py::other<Pln>())                                                                       \
      .def(py::self <= py::other<Drv>())                                                                       \
      .def(py::self <= py::other<Tnv>())                                                                       \
      .def(py::self <= py::other<Drb>())                                                                       \
      .def(py::self <= py::other<Tnb>())                                                                       \
      .def(py::self <= py::other<Drt>())                                                                       \
      .def(py::self <= py::other<Tnt>())                                                                       \
      .def(py::self <= py::other<Trs>())                                                                       \
      .def(py::self <= py::other<Mot>())                                                                       \
      .def(py::self <= py::other<Grt>())                                                                       \
      .def(py::self <= py::other<Trv>())                                                                       \
      .def(py::self <= py::other<Bst>())                                                                       \
      .def(py::self <= py::other<Con>())                                                                       \
      .def(py::self <= py::other<Dil>())                                                                       \
      .def(py::self <= py::other<Tsd>())                                                                       \
      .def(py::self <= py::other<ScaDrt>())                                                                    \
      .def(py::self - py::self)                                                                                \
      .def(py::self - py::other<Sca>())                                                                        \
      .def(py::self - py::other<Vec>())                                                                        \
      .def(py::self - py::other<Biv>())                                                                        \
      .def(py::self - py::other<Rot>())                                                                        \
      .def(py::self - py::other<Tri>())                                                                        \
      .def(py::self - py::other<Ori>())                                                                        \
      .def(py::self - py::other<Inf>())                                                                        \
      .def(py::self - py::other<Mnk>())                                                                        \
      .def(py::self - py::other<Pss>())                                                                        \
      .def(py::self - py::other<Pnt>())                                                                        \
      .def(py::self - py::other<Par>())                                                                        \
      .def(py::self - py::other<Cir>())                                                                        \
      .def(py::self - py::other<Sph>())                                                                        \
      .def(py::self - py::other<Dls>())                                                                        \
      .def(py::self - py::other<Flp>())                                                                        \
      .def(py::self - py::other<Dll>())                                                                        \
      .def(py::self - py::other<Lin>())                                                                        \
      .def(py::self - py::other<Dlp>())                                                                        \
      .def(py::self - py::other<Pln>())                                                                        \
      .def(py::self - py::other<Drv>())                                                                        \
      .def(py::self - py::other<Tnv>())                                                                        \
      .def(py::self - py::other<Drb>())                                                                        \
      .def(py::self - py::other<Tnb>())                                                                        \
      .def(py::self - py::other<Drt>())                                                                        \
      .def(py::self - py::other<Tnt>())                                                                        \
      .def(py::self - py::other<Trs>())                                                                        \
      .def(py::self - py::other<Mot>())                                                                        \
      .def(py::self - py::other<Grt>())                                                                        \
      .def(py::self - py::other<Trv>())                                                                        \
      .def(py::self - py::other<Bst>())                                                                        \
      .def(py::self - py::other<Con>())                                                                        \
      .def(py::self - py::other<Dil>())                                                                        \
      .def(py::self - py::other<Tsd>())                                                                        \
      .def(py::self - py::other<ScaDrt>())                                                                     \
      .def(py::self + py::self)                                                                                \
      .def(py::self + py::other<Sca>())                                                                        \
      .def(py::self + py::other<Vec>())                                                                        \
      .def(py::self + py::other<Biv>())                                                                        \
      .def(py::self + py::other<Rot>())                                                                        \
      .def(py::self + py::other<Tri>())                                                                        \
      .def(py::self + py::other<Ori>())                                                                        \
      .def(py::self + py::other<Inf>())                                                                        \
      .def(py::self + py::other<Mnk>())                                                                        \
      .def(py::self + py::other<Pss>())                                                                        \
      .def(py::self + py::other<Pnt>())                                                                        \
      .def(py::self + py::other<Par>())                                                                        \
      .def(py::self + py::other<Cir>())                                                                        \
      .def(py::self + py::other<Sph>())                                                                        \
      .def(py::self + py::other<Dls>())                                                                        \
      .def(py::self + py::other<Flp>())                                                                        \
      .def(py::self + py::other<Dll>())                                                                        \
      .def(py::self + py::other<Lin>())                                                                        \
      .def(py::self + py::other<Dlp>())                                                                        \
      .def(py::self + py::other<Pln>())                                                                        \
      .def(py::self + py::other<Drv>())                                                                        \
      .def(py::self + py::other<Tnv>())                                                                        \
      .def(py::self + py::other<Drb>())                                                                        \
      .def(py::self + py::other<Tnb>())                                                                        \
      .def(py::self + py::other<Drt>())                                                                        \
      .def(py::self + py::other<Tnt>())                                                                        \
      .def(py::self + py::other<Trs>())                                                                        \
      .def(py::self + py::other<Mot>())                                                                        \
      .def(py::self + py::other<Grt>())                                                                        \
      .def(py::self + py::other<Trv>())                                                                        \
      .def(py::self + py::other<Bst>())                                                                        \
      .def(py::self + py::other<Con>())                                                                        \
      .def(py::self + py::other<Dil>())                                                                        \
      .def(py::self + py::other<Tsd>())                                                                        \
      .def(py::self + py::other<ScaDrt>())                                                                     \
      .def(py::self ^ py::self)                                                                                \
      .def(py::self ^ py::other<Sca>())                                                                        \
      .def(py::self ^ py::other<Vec>())                                                                        \
      .def(py::self ^ py::other<Biv>())                                                                        \
      .def(py::self ^ py::other<Rot>())                                                                        \
      .def(py::self ^ py::other<Tri>())                                                                        \
      .def(py::self ^ py::other<Ori>())                                                                        \
      .def(py::self ^ py::other<Inf>())                                                                        \
      .def(py::self ^ py::other<Mnk>())                                                                        \
      .def(py::self ^ py::other<Pss>())                                                                        \
      .def(py::self ^ py::other<Pnt>())                                                                        \
      .def(py::self ^ py::other<Par>())                                                                        \
      .def(py::self ^ py::other<Cir>())                                                                        \
      .def(py::self ^ py::other<Sph>())                                                                        \
      .def(py::self ^ py::other<Dls>())                                                                        \
      .def(py::self ^ py::other<Flp>())                                                                        \
      .def(py::self ^ py::other<Dll>())                                                                        \
      .def(py::self ^ py::other<Lin>())                                                                        \
      .def(py::self ^ py::other<Dlp>())                                                                        \
      .def(py::self ^ py::other<Pln>())                                                                        \
      .def(py::self ^ py::other<Drv>())                                                                        \
      .def(py::self ^ py::other<Tnv>())                                                                        \
      .def(py::self ^ py::other<Drb>())                                                                        \
      .def(py::self ^ py::other<Tnb>())                                                                        \
      .def(py::self ^ py::other<Drt>())                                                                        \
      .def(py::self ^ py::other<Tnt>())                                                                        \
      .def(py::self ^ py::other<Trs>())                                                                        \
      .def(py::self ^ py::other<Mot>())                                                                        \
      .def(py::self ^ py::other<Grt>())                                                                        \
      .def(py::self ^ py::other<Trv>())                                                                        \
      .def(py::self ^ py::other<Bst>())                                                                        \
      .def(py::self ^ py::other<Con>())                                                                        \
      .def(py::self ^ py::other<Dil>())                                                                        \
      .def(py::self ^ py::other<Tsd>())                                                                        \
      .def(py::self ^ py::other<ScaDrt>())                                                                     \
      .def(py::self *py::self)                                                                                 \
      .def(py::self *py::other<Sca>())                                                                         \
      .def(py::self *py::other<Vec>())                                                                         \
      .def(py::self *py::other<Biv>())                                                                         \
      .def(py::self *py::other<Rot>())                                                                         \
      .def(py::self *py::other<Tri>())                                                                         \
      .def(py::self *py::other<Ori>())                                                                         \
      .def(py::self *py::other<Inf>())                                                                         \
      .def(py::self *py::other<Mnk>())                                                                         \
      .def(py::self *py::other<Pss>())                                                                         \
      .def(py::self *py::other<Pnt>())                                                                         \
      .def(py::self *py::other<Par>())                                                                         \
      .def(py::self *py::other<Cir>())                                                                         \
      .def(py::self *py::other<Sph>())                                                                         \
      .def(py::self *py::other<Dls>())                                                                         \
      .def(py::self *py::other<Flp>())                                                                         \
      .def(py::self *py::other<Dll>())                                                                         \
      .def(py::self *py::other<Lin>())                                                                         \
      .def(py::self *py::other<Dlp>())                                                                         \
      .def(py::self *py::other<Pln>())                                                                         \
      .def(py::self *py::other<Drv>())                                                                         \
      .def(py::self *py::other<Tnv>())                                                                         \
      .def(py::self *py::other<Drb>())                                                                         \
      .def(py::self *py::other<Tnb>())                                                                         \
      .def(py::self *py::other<Drt>())                                                                         \
      .def(py::self *py::other<Tnt>())                                                                         \
      .def(py::self *py::other<Trs>())                                                                         \
      .def(py::self *py::other<Mot>())                                                                         \
      .def(py::self *py::other<Grt>())                                                                         \
      .def(py::self *py::other<Trv>())                                                                         \
      .def(py::self *py::other<Bst>())                                                                         \
      .def(py::self *py::other<Con>())                                                                         \
      .def(py::self *py::other<Dil>())                                                                         \
      .def(py::self *py::other<Tsd>())                                                                         \
      .def(py::self *py::other<ScaDrt>())                                                                      \
      .def(py::self / py::self)                                                                                \
      .def(py::self / py::other<Sca>())                                                                        \
      .def(py::self / py::other<Vec>())                                                                        \
      .def(py::self / py::other<Biv>())                                                                        \
      .def(py::self / py::other<Rot>())                                                                        \
      .def(py::self / py::other<Tri>())                                                                        \
      .def(py::self / py::other<Ori>())                                                                        \
      .def(py::self / py::other<Inf>())                                                                        \
      .def(py::self / py::other<Mnk>())                                                                        \
      .def(py::self / py::other<Pss>())                                                                        \
      .def(py::self / py::other<Pnt>())                                                                        \
      .def(py::self / py::other<Par>())                                                                        \
      .def(py::self / py::other<Cir>())                                                                        \
      .def(py::self / py::other<Sph>())                                                                        \
      .def(py::self / py::other<Dls>())                                                                        \
      .def(py::self / py::other<Flp>())                                                                        \
      .def(py::self / py::other<Dll>())                                                                        \
      .def(py::self / py::other<Lin>())                                                                        \
      .def(py::self / py::other<Dlp>())                                                                        \
      .def(py::self / py::other<Pln>())                                                                        \
      .def(py::self / py::other<Drv>())                                                                        \
      .def(py::self / py::other<Tnv>())                                                                        \
      .def(py::self / py::other<Drb>())                                                                        \
      .def(py::self / py::other<Tnb>())                                                                        \
      .def(py::self / py::other<Drt>())                                                                        \
      .def(py::self / py::other<Tnt>())                                                                        \
      .def(py::self / py::other<Trs>())                                                                        \
      .def(py::self / py::other<Mot>())                                                                        \
      .def(py::self / py::other<Grt>())                                                                        \
      .def(py::self / py::other<Trv>())                                                                        \
      .def(py::self / py::other<Bst>())                                                                        \
      .def(py::self / py::other<Con>())                                                                        \
      .def(py::self / py::other<Dil>())                                                                        \
      .def(py::self / py::other<Tsd>())                                                                        \
      .def(py::self / py::other<ScaDrt>())                                                                     \
      .add_property("num", &MULTIVECTOR::get_num_bases)                                                        \
      .def("__str__", &MULTIVECTOR::to_string) __VA_ARGS__

PYBIND11_PLUGIN(libversor) {

  py::module m("libversor", "versor plugin");

  GAME_VSR_WRAP_MULTIVECTOR(Sca, .def(py::init<double>()));
  GAME_VSR_WRAP_MULTIVECTOR(
      Vec, .def(py::init<double, double, double>()).def("null", &Vec::null));
  GAME_VSR_WRAP_MULTIVECTOR(Biv, .def(py::init<double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Rot,
                            .def(py::init<double, double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Tri, .def(py::init<double>()));

  GAME_VSR_WRAP_MULTIVECTOR(Ori, .def(py::init<double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Inf, .def(py::init<double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Mnk, .def(py::init<double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Pss, .def(py::init<double>()));

  GAME_VSR_WRAP_MULTIVECTOR(
      Pnt, .def(py::init<double, double, double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Par);
  GAME_VSR_WRAP_MULTIVECTOR(Cir);
  GAME_VSR_WRAP_MULTIVECTOR(
      Sph, .def(py::init<double, double, double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(
      Dls, .def(py::init<double, double, double, double, double>()));

  GAME_VSR_WRAP_MULTIVECTOR(Flp,
                            .def(py::init<double, double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(
      Dll, .def(py::init<double, double, double, double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(
      Lin, .def(py::init<double, double, double, double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Dlp,
                            .def(py::init<double, double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Pln,
                            .def(py::init<double, double, double, double>()));

  GAME_VSR_WRAP_MULTIVECTOR(Drv, .def(py::init<double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Tnv, .def(py::init<double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Drb, .def(py::init<double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Tnb, .def(py::init<double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Drt, .def(py::init<double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Tnt, .def(py::init<double>()));

  GAME_VSR_WRAP_MULTIVECTOR(Trs,
                            .def(py::init<double, double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Mot,
                            .def(py::init<double, double, double, double,
                                          double, double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(
      Grt,
      .def(py::init<double, double, double, double, double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Trv);
  GAME_VSR_WRAP_MULTIVECTOR(Bst);
  GAME_VSR_WRAP_MULTIVECTOR(Con);
  GAME_VSR_WRAP_MULTIVECTOR(Dil);
  GAME_VSR_WRAP_MULTIVECTOR(Tsd);
  GAME_VSR_WRAP_MULTIVECTOR(ScaDrt, .def(py::init<double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(TrfTnv);
  GAME_VSR_WRAP_MULTIVECTOR(VecTri);

  return m.ptr();
}

#undef GAME_VSR_WRAP_MULTIVECTOR

} // namespace python

} // namespace vsr
