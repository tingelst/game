#ifndef GAME_GAME_VSR_CGA_TYPES_H_
#define GAME_GAME_VSR_CGA_TYPES_H_

#include "game/vsr/multivector.h"

namespace vsr
{

namespace cga
{

template <typename T>
using ConformalGeometricAlgebra = vsr::algebra<vsr::metric<4, 1, true>, T>;

template <typename T>
using Scalar = GASca<ConformalGeometricAlgebra<T>>;
template <typename T>
using Vector = GAVec<ConformalGeometricAlgebra<T>>;
template <typename T>
using Bivector = GABiv<ConformalGeometricAlgebra<T>>;
template <typename T>
using Trivector = GATri<ConformalGeometricAlgebra<T>>;
template <typename T>
using Rotor = GARot<ConformalGeometricAlgebra<T>>;

template <typename T>
using Origin = GAOri<ConformalGeometricAlgebra<T>>;
template <typename T>
using Infinity = GAInf<ConformalGeometricAlgebra<T>>;
template <typename T>
using Minkowski = GAMnk<ConformalGeometricAlgebra<T>>;
template <typename T>
using Pseudoscalar = GAPss<ConformalGeometricAlgebra<T>>;

template <typename T>
using Point = GAPnt<ConformalGeometricAlgebra<T>>;
template <typename T>
using PointPair = GAPar<ConformalGeometricAlgebra<T>>;
template <typename T>
using Circle = GACir<ConformalGeometricAlgebra<T>>;
template <typename T>
using Sphere = GASph<ConformalGeometricAlgebra<T>>;
template <typename T>
using DualSphere = GADls<ConformalGeometricAlgebra<T>>;

template <typename T>
using FlatPoint = GAFlp<ConformalGeometricAlgebra<T>>;
template <typename T>
using DualLine = GADll<ConformalGeometricAlgebra<T>>;
template <typename T>
using Line = GALin<ConformalGeometricAlgebra<T>>;
template <typename T>
using DualPlane = GADlp<ConformalGeometricAlgebra<T>>;
template <typename T>
using Plane = GAPln<ConformalGeometricAlgebra<T>>;

template <typename T>
using DirectionVector = GADrv<ConformalGeometricAlgebra<T>>;
template <typename T>
  using TangentVector = decltype(Point<T>{} ^ (-Point<T>{} <= (Vector<T>{} * Infinity<T>{})));
template <typename T>
using DirectionBivector = GADrb<ConformalGeometricAlgebra<T>>;
template <typename T>
using TangentBivector = GATnb<ConformalGeometricAlgebra<T>>;
template <typename T>
using DirectionTrivector = GADrt<ConformalGeometricAlgebra<T>>;
template <typename T>
using TangentTrivector = GATnt<ConformalGeometricAlgebra<T>>;

template <typename T>
using Translator = GATrs<ConformalGeometricAlgebra<T>>;
template <typename T>
using Motor = GAMot<ConformalGeometricAlgebra<T>>;
template <typename T>
using GeneralRotor = GAGrt<ConformalGeometricAlgebra<T>>;
template <typename T>
using Transversor = GATrv<ConformalGeometricAlgebra<T>>;
template <typename T>
using Boost = GABst<ConformalGeometricAlgebra<T>>;
template <typename T>
using ConformalRotor = GACon<ConformalGeometricAlgebra<T>>;
template <typename T>
using Dilator = GADil<ConformalGeometricAlgebra<T>>;
template <typename T>
using TranslatedDilator = GATsd<ConformalGeometricAlgebra<T>>;

using Sca = Scalar<double>;
using Vec = Vector<double>;
using Biv = Bivector<double>;
using Rot = Rotor<double>;
using Tri = Trivector<double>;

using Ori = Origin<double>;
using Inf = Infinity<double>;
using Mnk = Minkowski<double>;
using Pss = Pseudoscalar<double>;

using Pnt = Point<double>;
using Par = PointPair<double>;
using Cir = Circle<double>;
using Sph = Sphere<double>;
using Dls = DualSphere<double>;

using Flp = FlatPoint<double>;
using Dll = DualLine<double>;
using Lin = Line<double>;
using Dlp = DualPlane<double>;
using Pln = Plane<double>;

using Drv = DirectionVector<double>;
using Tnv = TangentVector<double>;
using Drb = DirectionBivector<double>;
using Tnb = TangentBivector<double>;
using Drt = DirectionTrivector<double>;
using Tnt = TangentTrivector<double>;

using Trs = Translator<double>;
using Mot = Motor<double>;
using Grt = GeneralRotor<double>;
using Trv = Transversor<double>;
using Bst = Boost<double>;
using Con = ConformalRotor<double>;
using Dil = Dilator<double>;
using Tsd = TranslatedDilator<double>;

} // namespace cga

} // namespace vsr

#endif  // GAME_GAME_VSR_CGA_TYPES_H_
