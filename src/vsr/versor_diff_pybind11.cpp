#include <pybind11/pybind11.h>

#include "game/vsr/cga_op.h"
#include "game/vsr/cga_types.h"

namespace vsr {

namespace python {

namespace py = pybind11;
using namespace vsr::cga;

void AddMultivectorDiff(py::module &m) {
  m.def("valkenburg", [](const Dll &a, const Dll &b) {
    auto two_b = b * 2.0;
    auto bb_inv = !(b * b);
    auto ret =
        Mot(a) +
        (Sca(1.0) - two_b - Gen::motor(-two_b)) * bb_inv * Dll(b * a) * 0.5;
    std::cout << ret << std::endl;
    return ret;
  });

  m.def("valkenburg2", [](const Biv &a, const Biv &b) {
    auto two_b = b * 2.0;
    auto bb_inv = !(b * b);
    auto ret = Rot(a +
                   (Sca(1.0) - two_b - Gen::rotor(-two_b)) * bb_inv *
                       Biv(b * a) * 0.5);
    return ret;
  });

  m.def("diff_rotor", [](const Rot &rot, const Biv &biv, const Vec &vec) {
    return (biv * Gen::log(rot) * rot * vec * ~rot)[0];
  });
}
}
}
