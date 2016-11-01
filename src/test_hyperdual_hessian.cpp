#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <game/hyperdual.h>

#include <game/vsr/cga_op.h>
#include <game/vsr/generic_op.h>

#include <iostream>

// using namespace game;
using namespace vsr::cga;
namespace py = pybind11;

struct VectorCorrespondencesCostFunctor {
  VectorCorrespondencesCostFunctor(const Vec &a, const Vec &b) : a_(a), b_(b) {}

  template <typename T>
  auto operator()(const T *const rotor, T *residual) const -> bool {
    Rotor<T> R(rotor);
    Vector<T> a(a_);
    Vector<T> b(b_);
    Vector<T> c = a.spin(R);

    for (int i = 0; i < 3; ++i) {
      residual[i] = c[i] - b[i];
    }

    return true;
  }

private:
  const Vec a_;
  const Vec b_;
};

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

PYBIND11_PLUGIN(hyperdual) {
  py::module m("hyperdual", "hyperdual");

  m.def("xpow3", [](const double &x) {
    hyperdual xh(x, 1, 1, 0);
    hyperdual ans = xh * xh * xh;
    return std::make_tuple(ans.real(), ans.eps1(), ans.eps2(), ans.eps1eps2());
  });

  m.def("hyperdualmotor", [](const Pnt &a, const Pnt &b, const Mot &M) {
    Point<hyperdual> ah(a);
    Point<hyperdual> bh(b);
    Motor<hyperdual> Mh(M);
    Vector<hyperdual> ch;
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
    ans = 0.5 * (ch * ch)[0];

    grad(0) = ans.eps1();
    H(0, 0) = ans.eps1eps2();

    Mhd[1].setvalues(0, 1, 0, 0);
    Mhd[2].setvalues(0, 0, 1, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(0, 1) = ans.eps1eps2();

    Mhd[1].setvalues(0, 1, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 1, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(0, 2) = ans.eps1eps2();

    Mhd[1].setvalues(0, 1, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 1, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(0, 3) = ans.eps1eps2();

    Mhd[1].setvalues(0, 1, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 1, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(0, 4) = ans.eps1eps2();

    Mhd[1].setvalues(0, 1, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 1, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(0, 5) = ans.eps1eps2();

    // New Row
    Mhd[1].setvalues(0, 0, 1, 0);
    Mhd[2].setvalues(0, 1, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    grad(1) = ans.eps1();
    H(1, 0) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 1, 1, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(1, 1) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 1, 0, 0);
    Mhd[3].setvalues(0, 0, 1, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(1, 2) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 1, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 1, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(1, 3) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 1, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 1, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(1, 4) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 1, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 1, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(1, 5) = ans.eps1eps2();

    // New Row
    Mhd[1].setvalues(0, 0, 1, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 1, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    grad(2) = ans.eps1();
    H(2, 0) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 1, 0);
    Mhd[3].setvalues(0, 1, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(2, 1) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 1, 1, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(2, 2) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 1, 0, 0);
    Mhd[4].setvalues(0, 0, 1, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(2, 3) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 1, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 1, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(2, 4) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 1, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 1, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(2, 5) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 1, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 1, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    grad(3) = ans.eps1();
    H(3, 0) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 1, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 1, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(3, 1) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 1, 0);
    Mhd[4].setvalues(0, 1, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(3, 2) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 1, 1, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(3, 3) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 1, 0, 0);
    Mhd[5].setvalues(0, 0, 1, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(3, 4) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 1, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 0, 1, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(3, 5) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 1, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 1, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    grad(4) = ans.eps1();
    H(4, 0) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 1, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 1, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(4, 1) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 1, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 1, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(4, 2) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 1, 0);
    Mhd[5].setvalues(0, 1, 0, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(4, 3) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 1, 1, 0);
    Mhd[6].setvalues(0, 0, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(4, 4) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 1, 0, 0);
    Mhd[6].setvalues(0, 0, 1, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(4, 5) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 1, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 1, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    grad(5) = ans.eps1();
    H(5, 0) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 1, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 1, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(5, 1) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 1, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 1, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(5, 2) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 1, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 1, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(5, 3) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 1, 0);
    Mhd[6].setvalues(0, 1, 0, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(5, 4) = ans.eps1eps2();

    Mhd[1].setvalues(0, 0, 0, 0);
    Mhd[2].setvalues(0, 0, 0, 0);
    Mhd[3].setvalues(0, 0, 0, 0);
    Mhd[4].setvalues(0, 0, 0, 0);
    Mhd[5].setvalues(0, 0, 0, 0);
    Mhd[6].setvalues(0, 1, 1, 0);
    ch = ah.spin(Mhd * Mh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(5, 5) = ans.eps1eps2();

    return std::make_tuple(ans.real(), grad, H);
  });

  m.def("hyperdualB", [](const Vec &a, const Vec &b, const Rot &R) {
    Vector<hyperdual> ah(a);
    Vector<hyperdual> bh(b);
    Rotor<hyperdual> Rh(R);
    Vector<hyperdual> ch;
    Rotor<hyperdual> Rhd;
    Rhd[0].setvalues(1, 0, 0, 0);
    Eigen::Vector3d grad;
    Eigen::Matrix3d J;
    Eigen::Matrix3d H;
    hyperdual ans;

    Rhd[1].setvalues(0, 1, 1, 0);
    Rhd[2].setvalues(0, 0, 0, 0);
    Rhd[3].setvalues(0, 0, 0, 0);
    ch = ah.spin(Rhd * Rh) - bh;
    ans = 0.5 * (ch * ch)[0];
    grad(0) = ans.eps1();
    H(0, 0) = ans.eps1eps2();

    Rhd[1].setvalues(0, 1, 0, 0);
    Rhd[2].setvalues(0, 0, 1, 0);
    Rhd[3].setvalues(0, 0, 0, 0);
    ch = ah.spin(Rhd * Rh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(0, 1) = ans.eps1eps2();

    Rhd[1].setvalues(0, 1, 0, 0);
    Rhd[2].setvalues(0, 0, 0, 0);
    Rhd[3].setvalues(0, 0, 1, 0);
    ch = ah.spin(Rhd * Rh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(0, 2) = ans.eps1eps2();

    Rhd[1].setvalues(0, 0, 1, 0);
    Rhd[2].setvalues(0, 1, 0, 0);
    Rhd[3].setvalues(0, 0, 0, 0);
    ch = ah.spin(Rhd * Rh) - bh;
    ans = 0.5 * (ch * ch)[0];
    grad(1) = ans.eps1();
    H(1, 0) = ans.eps1eps2();

    Rhd[1].setvalues(0, 0, 0, 0);
    Rhd[2].setvalues(0, 1, 1, 0);
    Rhd[3].setvalues(0, 0, 0, 0);
    ch = ah.spin(Rhd * Rh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(1, 1) = ans.eps1eps2();

    Rhd[1].setvalues(0, 0, 0, 0);
    Rhd[2].setvalues(0, 1, 0, 0);
    Rhd[3].setvalues(0, 0, 1, 0);
    ch = ah.spin(Rhd * Rh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(1, 2) = ans.eps1eps2();

    Rhd[1].setvalues(0, 0, 1, 0);
    Rhd[2].setvalues(0, 0, 0, 0);
    Rhd[3].setvalues(0, 1, 0, 0);
    ch = ah.spin(Rhd * Rh) - bh;
    ans = 0.5 * (ch * ch)[0];
    grad(2) = ans.eps1();
    H(2, 0) = ans.eps1eps2();

    Rhd[1].setvalues(0, 0, 0, 0);
    Rhd[2].setvalues(0, 0, 1, 0);
    Rhd[3].setvalues(0, 1, 0, 0);
    ch = ah.spin(Rhd * Rh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(2, 1) = ans.eps1eps2();

    Rhd[1].setvalues(0, 0, 0, 0);
    Rhd[2].setvalues(0, 0, 0, 0);
    Rhd[3].setvalues(0, 1, 1, 0);
    ch = ah.spin(Rhd * Rh) - bh;
    ans = 0.5 * (ch * ch)[0];
    H(2, 2) = ans.eps1eps2();

    return std::make_tuple(ans.real(), grad, H);
  });

  m.def("hyperdual1", [](const Vec &a, const Vec &b, const Rot &r) {
    Rotor<hyperdual> rh;
    rh[0].setvalues(r[0], 1, 0, 0);
    rh[1].setvalues(r[1], 0, 0, 0);
    rh[2].setvalues(r[2], 0, 0, 0);
    rh[3].setvalues(r[3], 0, 0, 0);
    Vector<hyperdual> ah(a);
    Vector<hyperdual> bh(b);
    Vector<hyperdual> ch = ah.spin(rh);
    ch[0].view();
    ch[1].view();
    ch[2].view();
  });

  m.def("hyperdual2", [](const Vec &a, const Vec &b, const Rot &r) {
    Rotor<hyperdual> rh;
    rh[0].setvalues(r[0], 0, 1, 0);
    rh[1].setvalues(r[1], 0, 0, 0);
    rh[2].setvalues(r[2], 0, 0, 0);
    rh[3].setvalues(r[3], 0, 0, 0);
    Vector<hyperdual> ah(a);
    Vector<hyperdual> bh(b);
    Vector<hyperdual> ch = ah.spin(rh);
    ch[0].view();
    ch[1].view();
    ch[2].view();
  });

  m.def("hyperdual12", [](const Vec &a, const Vec &b, const Rot &r) {
    Rotor<hyperdual> rh;
    rh[0].setvalues(r[0], 1, 0, 1);
    rh[1].setvalues(r[1], 0, 0, 0);
    rh[2].setvalues(r[2], 0, 0, 0);
    rh[3].setvalues(r[3], 0, 0, 0);
    Vector<hyperdual> ah(a);
    Vector<hyperdual> bh(b);
    Vector<hyperdual> ch = ah.spin(rh);
    ch[0].view();
    ch[1].view();
    ch[2].view();
  });

  return m.ptr();
}
