#include <boost/python.hpp>
#include <boost/numpy.hpp>

#include "game/versor_kinematic_calibration.h"

namespace bp = boost::python;
namespace np = boost::numpy;

BOOST_PYTHON_MODULE(libkinematic_calibration) {
  np::initialize();
  using namespace game::kinematic_calibration;

  bp::class_<KinematicCalibration>("KinematicCalibration",
                                   bp::init<const bp::dict&>())
      .def("run", &KinematicCalibration::Run)
      .def("summary", &KinematicCalibration::Summary)
      ;
}
