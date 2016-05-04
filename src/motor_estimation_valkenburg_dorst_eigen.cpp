#include "game/vsr/vsr.h"
#include <Eigen/Dense>

#include <iostream>

using vsr::cga::Gen;
using vsr::cga::Vec;
using vsr::cga::Biv;
using vsr::cga::Pnt;
using vsr::cga::Rot;
using vsr::cga::Mot;
using vsr::cga::Trs;
using vsr::cga::Lin;

const double kPi = 3.141592653589793238462643383279;

void EstimationRotor() {

  Rot rotor = Gen::rot(Biv{1.0, 1.0, 1.0}.unit() * kPi / 6.0);

  std::cout << rotor << std::endl;

  std::vector<Pnt> ps{Vec{1.0, 0.0, 0.0}.null(), Vec{0.0, 1.0, 0.0}.null(),
                      Vec{0.0, 0.0, 1.0}.null()};
  std::vector<Pnt> qs;
  for (const Pnt &p : ps) {
    qs.push_back(p.spin(rotor));
  }

  std::vector<Rot> rs;
  rs.reserve(4);

  for (int j = 0; j < ps.size(); ++j) {
    for (int i = 0; i < 4; ++i) {
      Rot ei{0.0, 0.0, 0.0, 0.0};
      ei[i] = 1.0;
      rs[i] = rs[i] + Rot(qs[j] * ei * ps[j]);
    }
  }

  Eigen::Matrix4d L = Eigen::Matrix4d::Zero();

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      Rot ei{0.0, 0.0, 0.0, 0.0};
      ei[i] = 1.0;
      L(i, j) = (~ei * rs[j])[0];
    }
  }

  // L(0, 0) = (~Rot(1.0, 0.0, 0.0, 0.0) * rs[0])[0];
  // L(1, 0) = (~Rot(0.0, 1.0, 0.0, 0.0) * rs[0])[0];
  // L(2, 0) = (~Rot(0.0, 0.0, 1.0, 0.0) * rs[0])[0];
  // L(3, 0) = (~Rot(0.0, 0.0, 0.0, 1.0) * rs[0])[0];
  // L(0, 1) = (~Rot(1.0, 0.0, 0.0, 0.0) * rs[1])[0];
  // L(1, 1) = (~Rot(0.0, 1.0, 0.0, 0.0) * rs[1])[0];
  // L(2, 1) = (~Rot(0.0, 0.0, 1.0, 0.0) * rs[1])[0];
  // L(3, 1) = (~Rot(0.0, 0.0, 0.0, 1.0) * rs[1])[0];
  // L(0, 2) = (~Rot(1.0, 0.0, 0.0, 0.0) * rs[2])[0];
  // L(1, 2) = (~Rot(0.0, 1.0, 0.0, 0.0) * rs[2])[0];
  // L(2, 2) = (~Rot(0.0, 0.0, 1.0, 0.0) * rs[2])[0];
  // L(3, 2) = (~Rot(0.0, 0.0, 0.0, 1.0) * rs[2])[0];
  // L(0, 3) = (~Rot(1.0, 0.0, 0.0, 0.0) * rs[3])[0];
  // L(1, 3) = (~Rot(0.0, 1.0, 0.0, 0.0) * rs[3])[0];
  // L(2, 3) = (~Rot(0.0, 0.0, 1.0, 0.0) * rs[3])[0];
  // L(3, 3) = (~Rot(0.0, 0.0, 0.0, 1.0) * rs[3])[0];

  std::cout << std::endl;
  std::cout << L << std::endl;

  Eigen::JacobiSVD<Eigen::Matrix4d> svd(L, Eigen::ComputeFullV);
  std::cout << svd.matrixV() << std::endl;
}

void EstimationMotor() {

  Trs translator = Gen::trs(Vec(1.0, 1.0, 1.0));
  Rot rotor = Gen::rot(Biv{1.0, 1.0, 1.0}.unit() * kPi / 6.0);
  Mot motor = translator * rotor;

  std::cout << motor << std::endl << std::endl;

  std::vector<Pnt> ps{Vec{1.0, 0.0, 0.0}.null(), Vec{0.0, 1.0, 0.0}.null(),
                      Vec{0.0, 0.0, 1.0}.null()};
  std::vector<Pnt> qs;
  for (const Pnt &p : ps) {
    qs.push_back(p.spin(motor));
  }

  std::vector<Lin> lps{
      vsr::cga::Construct::line(Vec(1, 2, 3), Vec(3, 4, 5)).unit(),
      vsr::cga::Construct::line(Vec(-1, 2, -3), Vec(1, 4, 5)).unit(),
      vsr::cga::Construct::line(Vec(0, 2, 3), Vec(3, 9, 5)).unit()};
  std::cout << lps[0] << std::endl;
  std::cout << lps[1] << std::endl;
  std::cout << lps[2] << std::endl;
  std::vector<Lin> lqs;
  for (const Lin &l : lps) {
    lqs.push_back(l.spin(motor));
  }

  using MotRec = vsr::Multivector<vsr::algebra<vsr::metric<4, 1, true>, double>,
                                  vsr::Basis<0, 3, 5, 6, 9, 10, 12, 15>>;
  std::vector<MotRec> rs;
  rs.reserve(8);

  for (int j = 0; j < ps.size(); ++j) {
    for (int i = 0; i < 8; ++i) {
      Mot ei{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      ei[i] = 1.0;
      // rs[i] = rs[i] + MotRec(qs[j] * ei * ps[j]);
      // rs[i] = rs[i] + MotRec(~lqs[j].dual() * ei * lps[j].dual());
      rs[i] = rs[i] + MotRec(-(~lqs[j]) * ei * lps[j]);
    }
  }

  Eigen::Matrix<double, 8, 8> L = Eigen::Matrix<double, 8, 8>::Zero();

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      Mot ei{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      ei[i] = 1.0;
      L(i, j) = (~ei * rs[j])[0];
    }
  }

  std::cout << std::endl;
  std::cout << L << std::endl;

  Eigen::Matrix4d Lrr = L.topLeftCorner(4, 4);
  Eigen::Matrix4d Lrq = L.topRightCorner(4, 4);
  Eigen::Matrix4d Lqr = L.bottomLeftCorner(4, 4);
  Eigen::Matrix4d Lqq = L.bottomRightCorner(4, 4);

  Eigen::Matrix4d Lp = Lrr;
    // - Lrq * (Lqq.inverse() * Lqr);

  std::cout << Lp << std::endl;

  Eigen::JacobiSVD<Eigen::Matrix<double, 4, 4>> svd(Lp, Eigen::ComputeFullV);
  Eigen::MatrixXd V = svd.matrixV();
  std::cout << V << std::endl;
  Eigen::MatrixXd r = V.col(3);
  std::cout << r << std::endl;

  Eigen::MatrixXd q = -(Lqq.inverse() * Lqr) * r;
  std::cout << q << std::endl;
}

int main() {

  EstimationMotor();

  return 0;
}
