#include <boost/python.hpp>
#include <boost/numpy.hpp>

#include "game/rigid_body_motion_estimation.h"

namespace bp = boost::python;
namespace np = boost::numpy;

BOOST_PYTHON_MODULE(librigid_body_motion_estimation) {
  np::initialize();

  bp::class_<game::RotorTranslationVectorEstimation>("RotorTranslationVectorEstimation")
      .def("run", &game::RotorTranslationVectorEstimation::Run)
      .def("summary", &game::RotorTranslationVectorEstimation::Summary);

  bp::class_<game::RotorTranslationVectorBivectorGeneratorEstimation>("RotorTranslationVectorBivectorGeneratorEstimation")
      .def("run", &game::RotorTranslationVectorBivectorGeneratorEstimation::Run)
      .def("summary", &game::RotorTranslationVectorBivectorGeneratorEstimation::Summary);

  bp::class_<game::GeneralRotorEstimation>("GeneralRotorEstimation")
      .def("run", &game::GeneralRotorEstimation::Run)
      .def("summary", &game::GeneralRotorEstimation::Summary);

}