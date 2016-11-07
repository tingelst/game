#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <game/hyperdual.h>

#include <game/vsr/cga_op.h>
#include <game/vsr/generic_op.h>

#include <iostream>

using namespace vsr::cga;
namespace py = pybind11;

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

PYBIND11_PLUGIN(hyperdual_lines) {
  py::module m("hyperdual_lines", "hyperdual_lines");
  m.def("lines", [](const Dll &a, const Dll &b, const Mot &M) {
    DualLine<hyperdual> ah(a);
    DualLine<hyperdual> bh(b);
    Motor<hyperdual> Mh(M);
    DualLine<hyperdual> ch;
    Motor<hyperdual> Mhd;
    Mhd[0].setvalues(1, 0, 0, 0);
    Mhd[7].setvalues(0, 0, 0, 0);
    Vector6d grad;
    Matrix6d H = Matrix6d::Zero(6, 6);
    hyperdual ans;

    Mhd[1].setvalues(0, 1, 1, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    // std::cout << a[0] << std::endl;
    // std::cout << a[1] << std::endl;
    // std::cout << a[2] << std::endl;
    // std::cout << a[3] << std::endl;
    // std::cout << a[4] << std::endl;
    // std::cout << a[5] << std::endl << std::endl;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);

    // ans = 0.5 * (Bivector<hyperdual>(ch) * ~Bivector<hyperdual>(ch))[0];
    // std::cout << ans << std::endl;
    grad(0) = ans.eps1();
    H(0, 0) = ans.eps1eps2();

    Mhd[1].setvalues(0, 1, 0, 0);
    Mhd[2].setvalues(0, 0, 1, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(0, 1) = ans.eps1eps2();

    Mhd[1].setvalues(0, 1, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 1, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(0, 2) = ans.eps1eps2();

    Mhd[1].setvalues(0, 1, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 1, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(0, 3) = ans.eps1eps2();

    Mhd[1].setvalues(0, 1, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 1, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(0, 4) = ans.eps1eps2();

    Mhd[1].setvalues(0, 1, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 1, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(0, 5) = ans.eps1eps2();

    // New Row
    Mhd[1].setvalues(0, 0, 1, 0);
    Mhd[2].setvalues(0, 1, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    grad(1) = ans.eps1();
    H(1, 0) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 1, 1, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(1, 1) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 1, 0, 0);
    Mhd[3].setvalues(0, 0, 1, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(1, 2) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 1, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 1, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(1, 3) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 1, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 1, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(1, 4) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 1, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 1, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(1, 5) = ans.eps1eps2();

    // New Row
    Mhd[1].setvalues(0, 0, 1, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 1, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    grad(2) = ans.eps1();
    H(2, 0) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 1, 0);
    Mhd[3].setvalues(0, 1, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(2, 1) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 1, 1, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(2, 2) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 1, 0, 0);
    Mhd[4].setvalues(0, 0, 1, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(2, 3) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 1, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 1, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(2, 4) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 1, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 1, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(2, 5) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 1, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 1, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    grad(3) = ans.eps1();
    H(3, 0) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 1, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 1, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(3, 1) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 1, 0);
    Mhd[4].setvalues(0, 1, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(3, 2) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 1, 1, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(3, 3) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 1, 0, 0);
    Mhd[5].setvalues(0, 0, 1, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(3, 4) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 1, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 1, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(3, 5) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 1, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 1, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    grad(4) = ans.eps1();
    H(4, 0) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 1, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 1, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(4, 1) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 1, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 1, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(4, 2) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 1, 0);
    Mhd[5].setvalues(0, 1, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(4, 3) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 1, 1, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(4, 4) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 1, 0, 0);
    Mhd[6].setvalues(0, 0, 1, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(4, 5) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 1, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 1, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    grad(5) = ans.eps1();
    H(5, 0) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 1, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 1, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(5, 1) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 1, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 1, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(5, 2) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 1, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 1, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(5, 3) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 1, 0);
    Mhd[6].setvalues(0, 1, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(5, 4) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 1, 1, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch[0] * ch[0] + ch[1] * ch[1] + ch[2] * ch[2] + ch[3] * ch[3] +
                 ch[4] * ch[4] + ch[5] * ch[5]);
    H(5, 5) = ans.eps1eps2();

    return std::make_tuple(ans.real(), grad, H);
  });

  return m.ptr();
}
