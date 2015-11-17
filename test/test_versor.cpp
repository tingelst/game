//
// Created by lars on 16.11.15.
//
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "game/vsr/cga_types.h"

TEST_CASE("conformal_types", "[conformal]") {
  using vsr::cga::Scalard;

  Scalard s{1.0};
  REQUIRE(s[0] == 1.0);

}
