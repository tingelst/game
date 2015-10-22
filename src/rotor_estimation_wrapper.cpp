#include <boost/python.hpp>
#include <boost/numpy.hpp>

#include "game/rotor_estimation.h"
#include "game/quaternion_estimation.h"
#include "game/rotor_estimation.h"
#include "game/rotor_bivector_generator_estimation.h"

namespace bp = boost::python;
namespace np = boost::numpy;

BOOST_PYTHON_MODULE(librotor_estimation) {
  np::initialize();
  bp::class_<game::RotorEstimation>("RotorEstimation", bp::init<bp::dict>())
      .def("run", &game::RotorEstimation::Run)
      .def("summary", &game::RotorEstimation::Summary)
      .def("set_solver_options", &game::RotorEstimation::SetSolverOptions);
  bp::class_<game::RotorBivectorGeneratorEstimation>(
      "RotorBivectorGeneratorEstimation")
      .def("run", &game::RotorBivectorGeneratorEstimation::Run)
      .def("summary", &game::RotorBivectorGeneratorEstimation::Summary);
  bp::class_<game::QuaternionEstimation>("QuaternionEstimation",
                                         bp::init<bp::dict>())
      .def("run", &game::QuaternionEstimation::Run)
      .def("summary", &game::QuaternionEstimation::Summary)
      .def("set_solver_options", &game::QuaternionEstimation::SetSolverOptions);
}
