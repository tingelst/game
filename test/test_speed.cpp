#include <cmath>
#include <iostream>
#include <hep/ga.hpp>
#include "game/types.h"

int main() {

  using namespace game;
  cga::Motor<double> m0t(1.0);
  m0t[0] = cos(M_PI / 6.0);
  m0t[1] = -sin(M_PI / 6.0);

  cga::Motor<double> m1t(1.0);
  cga::Motor<double> m2t(1.0);
  cga::Motor<double> m3t(1.0);
  cga::Motor<double> m4t(1.0);
  cga::Motor<double> m5t(1.0);

  auto mtest =
      hep::select<0, 3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23>(m0t * m1t * m2t *
                                                             m3t * m4t * m5t);

  cga::Point<double> pm{1.0, 2.0, 3.0, -6.5, 7.5};

  std::cout << pm[0] << std::endl;
  std::cout << pm[1] << std::endl;
  std::cout << pm[2] << std::endl;
  std::cout << pm[3] << std::endl;
  std::cout << pm[4] << std::endl;

//  auto pm_out = hep::eval(hep::grade<1>(m0t * m1t * m2t * m3t * m4t * m5t * pm * ~m0t *
//                              ~m1t * ~m2t * ~m3t * ~m4t * ~m5t));

//  std::cout << pm_out[0] << std::endl;
//    std::cout << pm_out[1] << std::endl;
//    std::cout << pm_out[2] << std::endl;
//    std::cout << pm_out[3] << std::endl;
//    std::cout << pm_out[4] << std::endl;

}
