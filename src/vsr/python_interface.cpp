
#include <boost/python.hpp>
#include <boost/numpy.hpp>

#include "game/vsr/cga_types.h"
#include "game/vsr/generic_op.h"

namespace vsr {

namespace python {

namespace bp = boost::python;
using namespace vsr::cga;

// Sca s;
//
//  game::versor::Vectord(game::versor::Vectord::*spin1)(const
//  game::versor::Rotord&) const =
//      &game::versor::Vectord::spin;
//
//  bp::class_<game::versor::Vectord>("Vector",
//                                    bp::init<double, double, double>())
//      .def(bp::init<game::versor::Vectord>())
//      .def("__getitem__", &game::versor::Vectord::at)
//      .def("unit", &game::versor::Vectord::unit)
//      .def("norm", &game::versor::Vectord::norm)
//      .def("print", &game::versor::Vectord::print)
//      .def("reverse", &game::versor::Vectord::reverse)
//      .def("inverse", &game::versor::Vectord::inverse)
//      .def("dual", &game::versor::Vectord::duale)
//      .def("null", &game::versor::Vectord::null)
//      .def("spin", spin1)
//      .add_property("num", &game::versor::Vectord::get_num_bases)
//      .def(bp::self | bp::other<game::versor::Rotord>())
//      .def(bp::self <= bp::self)
//      .def(bp::self + bp::self)
//      .def(bp::self ^ bp::self)
//      .def(bp::self * bp::self)
//      .def(bp::self / bp::self)
//      .def("__str__", &game::versor::Vectord::to_string);

#define GAME_VSR_WRAP_MULTIVECTOR(MULTIVECTOR, ...)                        \
  MULTIVECTOR (MULTIVECTOR::*MULTIVECTOR##SpinByRotor)(const Rot&) const = \
      &MULTIVECTOR::spin;                                                  \
  MULTIVECTOR (MULTIVECTOR::*MULTIVECTOR##SpinByMotor)(const Mot&) const = \
      &MULTIVECTOR::spin;                                                  \
  MULTIVECTOR (MULTIVECTOR::*MULTIVECTOR##SpinByGeneralRotor)(const Grt&)  \
      const = &MULTIVECTOR::spin;                                          \
  bp::class_<MULTIVECTOR>(#MULTIVECTOR)                                    \
      .def(bp::init<MULTIVECTOR>())                                        \
      .def("__getitem__", &MULTIVECTOR::at)                                \
      .def("unit", &MULTIVECTOR::unit)                                     \
      .def("runit", &MULTIVECTOR::runit)                                   \
      .def("tunit", &MULTIVECTOR::tunit)                                   \
      .def("wt", &MULTIVECTOR::wt)                                         \
      .def("rwt", &MULTIVECTOR::rwt)                                       \
      .def("norm", &MULTIVECTOR::norm)                                     \
      .def("rnorm", &MULTIVECTOR::rnorm)                                   \
      .def("rev", &MULTIVECTOR::reverse)                                   \
      .def("inv", &MULTIVECTOR::inverse)                                   \
      .def("duale", &MULTIVECTOR::duale)                                   \
      .def("unduale", &MULTIVECTOR::unduale)                               \
      .def("dual", &MULTIVECTOR::dual)                                     \
      .def("undual", &MULTIVECTOR::undual)                                 \
      .def("spin", MULTIVECTOR##SpinByRotor)                               \
      .def("spin", MULTIVECTOR##SpinByMotor)                               \
      .def("spin", MULTIVECTOR##SpinByGeneralRotor)                        \
      .def("__neg__", &MULTIVECTOR::negate)                                \
      .def(bp::self == bp::self)                                           \
      .def(bp::self* bp::other<double>())                                  \
      .def(bp::self / bp::other<double>())                                 \
      .def(bp::self + bp::other<double>())                                 \
      .def(bp::self <= bp::self)                                           \
      .def(bp::self <= bp::other<Sca>())                                   \
      .def(bp::self <= bp::other<Vec>())                                   \
      .def(bp::self <= bp::other<Biv>())                                   \
      .def(bp::self <= bp::other<Rot>())                                   \
      .def(bp::self <= bp::other<Tri>())                                   \
      .def(bp::self <= bp::other<Ori>())                                   \
      .def(bp::self <= bp::other<Inf>())                                   \
      .def(bp::self <= bp::other<Mnk>())                                   \
      .def(bp::self <= bp::other<Pss>())                                   \
      .def(bp::self <= bp::other<Pnt>())                                   \
      .def(bp::self <= bp::other<Par>())                                   \
      .def(bp::self <= bp::other<Cir>())                                   \
      .def(bp::self <= bp::other<Sph>())                                   \
      .def(bp::self <= bp::other<Dls>())                                   \
      .def(bp::self <= bp::other<Flp>())                                   \
      .def(bp::self <= bp::other<Dll>())                                   \
      .def(bp::self <= bp::other<Lin>())                                   \
      .def(bp::self <= bp::other<Dlp>())                                   \
      .def(bp::self <= bp::other<Pln>())                                   \
      .def(bp::self <= bp::other<Drv>())                                   \
      .def(bp::self <= bp::other<Tnv>())                                   \
      .def(bp::self <= bp::other<Drb>())                                   \
      .def(bp::self <= bp::other<Tnb>())                                   \
      .def(bp::self <= bp::other<Drt>())                                   \
      .def(bp::self <= bp::other<Tnt>())                                   \
      .def(bp::self <= bp::other<Trs>())                                   \
      .def(bp::self <= bp::other<Mot>())                                   \
      .def(bp::self <= bp::other<Grt>())                                   \
      .def(bp::self <= bp::other<Trv>())                                   \
      .def(bp::self <= bp::other<Bst>())                                   \
      .def(bp::self <= bp::other<Con>())                                   \
      .def(bp::self <= bp::other<Dil>())                                   \
      .def(bp::self <= bp::other<Tsd>())                                   \
      .def(bp::self - bp::self)                                            \
      .def(bp::self - bp::other<Sca>())                                    \
      .def(bp::self - bp::other<Vec>())                                    \
      .def(bp::self - bp::other<Biv>())                                    \
      .def(bp::self - bp::other<Rot>())                                    \
      .def(bp::self - bp::other<Tri>())                                    \
      .def(bp::self - bp::other<Ori>())                                    \
      .def(bp::self - bp::other<Inf>())                                    \
      .def(bp::self - bp::other<Mnk>())                                    \
      .def(bp::self - bp::other<Pss>())                                    \
      .def(bp::self - bp::other<Pnt>())                                    \
      .def(bp::self - bp::other<Par>())                                    \
      .def(bp::self - bp::other<Cir>())                                    \
      .def(bp::self - bp::other<Sph>())                                    \
      .def(bp::self - bp::other<Dls>())                                    \
      .def(bp::self - bp::other<Flp>())                                    \
      .def(bp::self - bp::other<Dll>())                                    \
      .def(bp::self - bp::other<Lin>())                                    \
      .def(bp::self - bp::other<Dlp>())                                    \
      .def(bp::self - bp::other<Pln>())                                    \
      .def(bp::self - bp::other<Drv>())                                    \
      .def(bp::self - bp::other<Tnv>())                                    \
      .def(bp::self - bp::other<Drb>())                                    \
      .def(bp::self - bp::other<Tnb>())                                    \
      .def(bp::self - bp::other<Drt>())                                    \
      .def(bp::self - bp::other<Tnt>())                                    \
      .def(bp::self - bp::other<Trs>())                                    \
      .def(bp::self - bp::other<Mot>())                                    \
      .def(bp::self - bp::other<Grt>())                                    \
      .def(bp::self - bp::other<Trv>())                                    \
      .def(bp::self - bp::other<Bst>())                                    \
      .def(bp::self - bp::other<Con>())                                    \
      .def(bp::self - bp::other<Dil>())                                    \
      .def(bp::self - bp::other<Tsd>())                                    \
      .def(bp::self + bp::self)                                            \
      .def(bp::self + bp::other<Sca>())                                    \
      .def(bp::self + bp::other<Vec>())                                    \
      .def(bp::self + bp::other<Biv>())                                    \
      .def(bp::self + bp::other<Rot>())                                    \
      .def(bp::self + bp::other<Tri>())                                    \
      .def(bp::self + bp::other<Ori>())                                    \
      .def(bp::self + bp::other<Inf>())                                    \
      .def(bp::self + bp::other<Mnk>())                                    \
      .def(bp::self + bp::other<Pss>())                                    \
      .def(bp::self + bp::other<Pnt>())                                    \
      .def(bp::self + bp::other<Par>())                                    \
      .def(bp::self + bp::other<Cir>())                                    \
      .def(bp::self + bp::other<Sph>())                                    \
      .def(bp::self + bp::other<Dls>())                                    \
      .def(bp::self + bp::other<Flp>())                                    \
      .def(bp::self + bp::other<Dll>())                                    \
      .def(bp::self + bp::other<Lin>())                                    \
      .def(bp::self + bp::other<Dlp>())                                    \
      .def(bp::self + bp::other<Pln>())                                    \
      .def(bp::self + bp::other<Drv>())                                    \
      .def(bp::self + bp::other<Tnv>())                                    \
      .def(bp::self + bp::other<Drb>())                                    \
      .def(bp::self + bp::other<Tnb>())                                    \
      .def(bp::self + bp::other<Drt>())                                    \
      .def(bp::self + bp::other<Tnt>())                                    \
      .def(bp::self + bp::other<Trs>())                                    \
      .def(bp::self + bp::other<Mot>())                                    \
      .def(bp::self + bp::other<Grt>())                                    \
      .def(bp::self + bp::other<Trv>())                                    \
      .def(bp::self + bp::other<Bst>())                                    \
      .def(bp::self + bp::other<Con>())                                    \
      .def(bp::self + bp::other<Dil>())                                    \
      .def(bp::self + bp::other<Tsd>())                                    \
      .def(bp::self ^ bp::self)                                            \
      .def(bp::self ^ bp::other<Sca>())                                    \
      .def(bp::self ^ bp::other<Vec>())                                    \
      .def(bp::self ^ bp::other<Biv>())                                    \
      .def(bp::self ^ bp::other<Rot>())                                    \
      .def(bp::self ^ bp::other<Tri>())                                    \
      .def(bp::self ^ bp::other<Ori>())                                    \
      .def(bp::self ^ bp::other<Inf>())                                    \
      .def(bp::self ^ bp::other<Mnk>())                                    \
      .def(bp::self ^ bp::other<Pss>())                                    \
      .def(bp::self ^ bp::other<Pnt>())                                    \
      .def(bp::self ^ bp::other<Par>())                                    \
      .def(bp::self ^ bp::other<Cir>())                                    \
      .def(bp::self ^ bp::other<Sph>())                                    \
      .def(bp::self ^ bp::other<Dls>())                                    \
      .def(bp::self ^ bp::other<Flp>())                                    \
      .def(bp::self ^ bp::other<Dll>())                                    \
      .def(bp::self ^ bp::other<Lin>())                                    \
      .def(bp::self ^ bp::other<Dlp>())                                    \
      .def(bp::self ^ bp::other<Pln>())                                    \
      .def(bp::self ^ bp::other<Drv>())                                    \
      .def(bp::self ^ bp::other<Tnv>())                                    \
      .def(bp::self ^ bp::other<Drb>())                                    \
      .def(bp::self ^ bp::other<Tnb>())                                    \
      .def(bp::self ^ bp::other<Drt>())                                    \
      .def(bp::self ^ bp::other<Tnt>())                                    \
      .def(bp::self ^ bp::other<Trs>())                                    \
      .def(bp::self ^ bp::other<Mot>())                                    \
      .def(bp::self ^ bp::other<Grt>())                                    \
      .def(bp::self ^ bp::other<Trv>())                                    \
      .def(bp::self ^ bp::other<Bst>())                                    \
      .def(bp::self ^ bp::other<Con>())                                    \
      .def(bp::self ^ bp::other<Dil>())                                    \
      .def(bp::self ^ bp::other<Tsd>())                                    \
      .def(bp::self* bp::self)                                             \
      .def(bp::self* bp::other<Sca>())                                     \
      .def(bp::self* bp::other<Vec>())                                     \
      .def(bp::self* bp::other<Biv>())                                     \
      .def(bp::self* bp::other<Rot>())                                     \
      .def(bp::self* bp::other<Tri>())                                     \
      .def(bp::self* bp::other<Ori>())                                     \
      .def(bp::self* bp::other<Inf>())                                     \
      .def(bp::self* bp::other<Mnk>())                                     \
      .def(bp::self* bp::other<Pss>())                                     \
      .def(bp::self* bp::other<Pnt>())                                     \
      .def(bp::self* bp::other<Par>())                                     \
      .def(bp::self* bp::other<Cir>())                                     \
      .def(bp::self* bp::other<Sph>())                                     \
      .def(bp::self* bp::other<Dls>())                                     \
      .def(bp::self* bp::other<Flp>())                                     \
      .def(bp::self* bp::other<Dll>())                                     \
      .def(bp::self* bp::other<Lin>())                                     \
      .def(bp::self* bp::other<Dlp>())                                     \
      .def(bp::self* bp::other<Pln>())                                     \
      .def(bp::self* bp::other<Drv>())                                     \
      .def(bp::self* bp::other<Tnv>())                                     \
      .def(bp::self* bp::other<Drb>())                                     \
      .def(bp::self* bp::other<Tnb>())                                     \
      .def(bp::self* bp::other<Drt>())                                     \
      .def(bp::self* bp::other<Tnt>())                                     \
      .def(bp::self* bp::other<Trs>())                                     \
      .def(bp::self* bp::other<Mot>())                                     \
      .def(bp::self* bp::other<Grt>())                                     \
      .def(bp::self* bp::other<Trv>())                                     \
      .def(bp::self* bp::other<Bst>())                                     \
      .def(bp::self* bp::other<Con>())                                     \
      .def(bp::self* bp::other<Dil>())                                     \
      .def(bp::self* bp::other<Tsd>())                                     \
      .def(bp::self / bp::self)                                            \
      .def(bp::self / bp::other<Sca>())                                    \
      .def(bp::self / bp::other<Vec>())                                    \
      .def(bp::self / bp::other<Biv>())                                    \
      .def(bp::self / bp::other<Rot>())                                    \
      .def(bp::self / bp::other<Tri>())                                    \
      .def(bp::self / bp::other<Ori>())                                    \
      .def(bp::self / bp::other<Inf>())                                    \
      .def(bp::self / bp::other<Mnk>())                                    \
      .def(bp::self / bp::other<Pss>())                                    \
      .def(bp::self / bp::other<Pnt>())                                    \
      .def(bp::self / bp::other<Par>())                                    \
      .def(bp::self / bp::other<Cir>())                                    \
      .def(bp::self / bp::other<Sph>())                                    \
      .def(bp::self / bp::other<Dls>())                                    \
      .def(bp::self / bp::other<Flp>())                                    \
      .def(bp::self / bp::other<Dll>())                                    \
      .def(bp::self / bp::other<Lin>())                                    \
      .def(bp::self / bp::other<Dlp>())                                    \
      .def(bp::self / bp::other<Pln>())                                    \
      .def(bp::self / bp::other<Drv>())                                    \
      .def(bp::self / bp::other<Tnv>())                                    \
      .def(bp::self / bp::other<Drb>())                                    \
      .def(bp::self / bp::other<Tnb>())                                    \
      .def(bp::self / bp::other<Drt>())                                    \
      .def(bp::self / bp::other<Tnt>())                                    \
      .def(bp::self / bp::other<Trs>())                                    \
      .def(bp::self / bp::other<Mot>())                                    \
      .def(bp::self / bp::other<Grt>())                                    \
      .def(bp::self / bp::other<Trv>())                                    \
      .def(bp::self / bp::other<Bst>())                                    \
      .def(bp::self / bp::other<Con>())                                    \
      .def(bp::self / bp::other<Dil>())                                    \
      .def(bp::self / bp::other<Tsd>())                                    \
      .add_property("num", &MULTIVECTOR::get_num_bases)                    \
      .def("__str__", &MULTIVECTOR::to_string) __VA_ARGS__

BOOST_PYTHON_MODULE(libversor) {
  GAME_VSR_WRAP_MULTIVECTOR(Sca, .def(bp::init<double>()));
  GAME_VSR_WRAP_MULTIVECTOR(
      Vec, .def(bp::init<double, double, double>()).def("null", &Vec::null));
  GAME_VSR_WRAP_MULTIVECTOR(Biv, .def(bp::init<double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Rot,
                            .def(bp::init<double, double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Tri, .def(bp::init<double>()));

  GAME_VSR_WRAP_MULTIVECTOR(Ori, .def(bp::init<double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Inf, .def(bp::init<double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Mnk, .def(bp::init<double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Pss, .def(bp::init<double>()));

  GAME_VSR_WRAP_MULTIVECTOR(
      Pnt, .def(bp::init<double, double, double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Par);
  GAME_VSR_WRAP_MULTIVECTOR(Cir);
  GAME_VSR_WRAP_MULTIVECTOR(
      Sph, .def(bp::init<double, double, double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(
      Dls, .def(bp::init<double, double, double, double, double>()));

  GAME_VSR_WRAP_MULTIVECTOR(Flp,
                            .def(bp::init<double, double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(
      Dll, .def(bp::init<double, double, double, double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(
      Lin, .def(bp::init<double, double, double, double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Dlp,
                            .def(bp::init<double, double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Pln,
                            .def(bp::init<double, double, double, double>()));

  GAME_VSR_WRAP_MULTIVECTOR(Drv, .def(bp::init<double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Tnv, .def(bp::init<double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Drb, .def(bp::init<double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Tnb, .def(bp::init<double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Drt, .def(bp::init<double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Tnt, .def(bp::init<double>()));

  GAME_VSR_WRAP_MULTIVECTOR(Trs,
                            .def(bp::init<double, double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(
      Mot, .def(bp::init<double, double, double, double, double, double, double,
                         double>()));
  GAME_VSR_WRAP_MULTIVECTOR(
      Grt,
      .def(bp::init<double, double, double, double, double, double, double>()));
  GAME_VSR_WRAP_MULTIVECTOR(Trv);
  GAME_VSR_WRAP_MULTIVECTOR(Bst);
  GAME_VSR_WRAP_MULTIVECTOR(Con);
  GAME_VSR_WRAP_MULTIVECTOR(Dil);
  GAME_VSR_WRAP_MULTIVECTOR(Tsd);
}

#undef GAME_VSR_WRAP_MULTIVECTOR

}  // namespace python

}  // namespace vsr
