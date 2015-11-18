//
// Created by lars on 16.11.15.
//
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "game/vsr/cga_types.h"
#include "game/vsr/generic_op.h"

#include "game/motor_parameterization.h"

TEST_CASE("motor_from_dual_line", "[bivector_generator]") {

  double x[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double delta[6] = {0.0, -0.26, 0.0, -0.62, -0.5, -0.36};
  double x_plus_delta[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  game::MotorFromBivectorGenerator m;

  m(x, delta, x_plus_delta);


  std::cout << "[ ";
  for (int i = 0; i < 8; ++i) {
    std::cout << x_plus_delta[i] << " ";
  }
  std::cout << " ]" << std::endl;


}

TEST_CASE("vector_project_onto_bivector", "[bivector_generator]") {

  using vsr::cga::Vec;
  using vsr::cga::Biv;


  auto vsr_v = Vec(1,2,3);
  auto vsr_b = Biv(1,1,1);

  double v[3] = {1.0, 2.0, 3.0};
  double b[3] = {1.0, 1.0, 1.0};
  double bn[3] = {1.0, 1.0, 1.0};
  double res[3] = {0.0,0.0,0.0};

  game::Normalize3(b, bn);
  auto vsr_res = vsr::nga::Op::project(vsr_v, vsr_b);
  game::ProjectVectorOntoBivector(v, bn, res);

  REQUIRE(vsr_res[0] == Approx(res[0]).epsilon(10e-8));
  REQUIRE(vsr_res[1] == Approx(res[1]).epsilon(10e-8));
  REQUIRE(vsr_res[2] == Approx(res[2]).epsilon(10e-8));

  vsr_res = vsr::nga::Op::reject(vsr_v, vsr_b);
  game::RejectVectorFromBivector(v,bn,res);

  REQUIRE(vsr_res[0] == Approx(res[0]).epsilon(10e-8));
  REQUIRE(vsr_res[1] == Approx(res[1]).epsilon(10e-8));
  REQUIRE(vsr_res[2] == Approx(res[2]).epsilon(10e-8));

}
