#ifndef GAME_GAME_TYPES_H_
#define GAME_GAME_TYPES_H_

namespace cga {
template <typename T, int... components>
using Multivector =
    hep::multi_vector<hep::algebra<T, 4, 1>, hep::list<components...>>;

template <typename T>
using Scalar = Multivector<T, 0>;
template <typename T>
using Infty = cga::Multivector<T, 8, 16>;
template <typename T>
using Orig = cga::Multivector<T, 8, 16>;
template <typename T>
using EuclideanPoint = cga::Multivector<T, 1, 2, 4>;
}

#endif  // GAME_GAME_TYPES_H_
