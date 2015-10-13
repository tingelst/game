#ifndef GAME_GAME_TYPES_H_
#define GAME_GAME_TYPES_H_

namespace cga {
template <typename T, int... components>
using Multivector =
    hep::multi_vector<hep::algebra<T, 4, 1>, hep::list<components...>>;

//      using cga = hep::algebra<T, 4, 1>;
//      using Scalar = hep::multi_vector<cga, hep::list<0> >;
//      using PseudoScalarE3 = hep::multi_vector<cga, hep::list<7> >;
//      using VectorE3 = hep::multi_vector<cga, hep::list<1, 2, 4> >;
//      using PointE3 = hep::multi_vector<cga, hep::list<1, 2, 4> >;
//      using Infty = hep::multi_vector<cga, hep::list<8, 16> >;
//      using Orig = hep::multi_vector<cga, hep::list<8, 16> >;
//      using Point = hep::multi_vector<cga, hep::list<1, 2, 4, 8, 16> >;
//      using Rotor = hep::multi_vector<cga, hep::list<0, 3, 5, 6> >;
//      using Motor = hep::multi_vector<
//          cga, hep::list<0, 3, 5, 6, 9, 10, 12, 15, 17, 18, 20, 23> >;
//      using E1 = hep::multi_vector<cga, hep::list<1> >;
//      using E2 = hep::multi_vector<cga, hep::list<2> >;
//      using E3 = hep::multi_vector<cga, hep::list<4> >;
//      using Ep = hep::multi_vector<cga, hep::list<8> >;
//      using Em = hep::multi_vector<cga, hep::list<16> >;

template <typename T>
using E1 = Multivector<T, 1>;
template <typename T>
using E2 = Multivector<T, 2>;
template <typename T>
using E3 = Multivector<T, 4>;

template <typename T>
using Scalar = Multivector<T, 0>;
template <typename T>
using Infty = Multivector<T, 8, 16>;
template <typename T>
using Orig = Multivector<T, 8, 16>;

template <typename T>
using Point = Multivector<T, 1, 2, 4, 8, 16>;

template <typename T>
using Rotor = Multivector<T, 0, 3, 5, 6>;
template <typename T>
using Translator = Multivector<T, 0, 9, 10, 12, 17, 18, 20>;
template <typename T>
using GeneralRotor = Multivector<T, 0, 3, 5, 6, 9, 10, 12, 17, 18, 20>;
template <typename T>
using Motor = Multivector<T,0,3,5,6,9,10,12,15,17,18,20,23>;

template <typename T>
using Vector = Multivector<T, 1, 2, 4>;
template <typename T>
using EuclideanVector = Vector<T>;
template <typename T>
using EuclideanPoint = EuclideanVector<T>;
}

#endif  // GAME_GAME_TYPES_H_
