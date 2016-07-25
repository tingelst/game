
#include "game/vsr/generic_op.h"
#include <iostream>

using ga63 = vsr::algebra<vsr::metric<6, 3, false>, double>;
using A = vsr::Multivector<ga63, vsr::Basis<1, 2, 3>>;
using B = vsr::Multivector<ga63, vsr::Basis<1, 11, 32, 309>>;
using C = decltype(A(0, 0, 0, 0) * B(0, 0, 0, 0));

using ga33 = vsr::algebra<vsr::metric<3, 3, false>, double>;

int main() {
  std::cout << "GA 63" << std::endl;

  A a(1.0, 2.0, 3.0);
  B b(1, 2, 3, 4);

  (a * b).print();

  return 0;
}
