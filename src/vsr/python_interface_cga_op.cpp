
#include <boost/python.hpp>
#include <boost/numpy.hpp>

#include "game/vsr/cga_types.h"
#include "game/vsr/cga_op.h"

namespace vsr {

namespace python {

namespace bp = boost::python;
using namespace vsr::cga;



BOOST_PYTHON_MODULE_INIT(libversor_cga_op) {

bp::class_<Gen>("Gen")
  //.def("log", (Dll(Gen::*)(const Mot&))&Gen::log).staticmethod("log")
  .def("log", &Gen::logMotor).staticmethod("log")
  .def("mot", &Gen::mot).staticmethod("mot")
  ;


}


}  // namespace python

}  // namespace vsr
