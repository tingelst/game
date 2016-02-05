#include <pybind11/pybind11.h>
#include "game/vsr/cga_types.h"
#include "game/vsr/generic_op.h"
namespace py = pybind11;
using namespace vsr::cga;
PYBIND11_PLUGIN(versorninja) { 
  py::module m("versorninja");
  py::class_<Dil>(m, "Dil")

  ;
  py::class_<Par>(m, "Par")

  ;
  py::class_<Tsd>(m, "Tsd")

  ;
  py::class_<Grt>(m, "Grt")

  ;
  py::class_<Dll>(m, "Dll")

  ;
  py::class_<Dls>(m, "Dls")

  ;
  py::class_<Dlp>(m, "Dlp")

  ;
  py::class_<Vec>(m, "Vec")

  ;
  py::class_<Bst>(m, "Bst")

  ;
  py::class_<Pnt>(m, "Pnt")

  ;
  py::class_<Mot>(m, "Mot")

  ;
  py::class_<Ori>(m, "Ori")

  ;
  py::class_<Biv>(m, "Biv")

  ;
  py::class_<Pln>(m, "Pln")

  ;
  py::class_<Con>(m, "Con")

  ;
  py::class_<Lin>(m, "Lin")

  ;
  py::class_<Tri>(m, "Tri")

  ;
  py::class_<Tnt>(m, "Tnt")

  ;
  py::class_<Tnv>(m, "Tnv")

  ;
  py::class_<Flp>(m, "Flp")

  ;
  py::class_<Drt>(m, "Drt")

  ;
  py::class_<Drv>(m, "Drv")

  ;
  py::class_<Cir>(m, "Cir")

  ;
  py::class_<Tnb>(m, "Tnb")

  ;
  py::class_<Trs>(m, "Trs")

  ;
  py::class_<Drb>(m, "Drb")

  ;
  py::class_<Trv>(m, "Trv")

  ;
  py::class_<Sca>(m, "Sca")

  ;
  py::class_<Sph>(m, "Sph")

  ;
  py::class_<Mnk>(m, "Mnk")

  ;
  py::class_<Pss>(m, "Pss")

  ;
  py::class_<Inf>(m, "Inf")

  ;
  py::class_<Rot>(m, "Rot")

  ;

  return m.ptr(); 
}