/*!  @file 3D CGA Operations on Multivectors of @ref round

 *
 =====================================================================================
 *
 *       Filename:  vsr_cga3D_round.h
 *
 *    Description:  operations on rounds specialized for 3D
 *
 *        Version:  1.0
 *        Created:  07/13/2015 18:52:36
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Pablo Colapinto (), gmail->wolftype
 *   Organization:  wolftype
 *
 *
 =====================================================================================
 */

#ifndef vsr_cga3D_round_INC
#define vsr_cga3D_round_INC

#include "game/vsr/generic_op.h"
#include "game/vsr/cga_types.h"

namespace vsr {
namespace cga {

/*!
 * 3D operations on @ref round types (Pnts, Pnt Pars, Cirs, Sphs)
   @ingroup cgaops


    3D CGA Template Specializations of ND functions found in vsr::nga::Round
    Note: the ND functions are inlined and can be used instead, but using these
    will speed up compile times since they are precompiled into libvsr.a

    @sa @ref round for a list of @ref cgatypes on which these methods operate
    @sa vsr::nga::Round for the @ref generic n-dimensional implementation
 details

 */
struct Round {
  ///@{
  /*! Null Pnt from a vec
   */
  static Pnt null(const Vec& v);

  /*! Null Pnt from a Pnt
   */
  static Pnt null(const Pnt& v);

  /*! Or Null Pnt from Coordinates (x,y,z,...)
   */
  static Pnt null(VSR_PRECISION x, VSR_PRECISION y, VSR_PRECISION z);

  /*! Null Pnt from Coordinates
  */
  static Pnt point(VSR_PRECISION x, VSR_PRECISION y, VSR_PRECISION z);
  ///@}

  ///@{
  /*! Dual Sph from Coordinate Center

       note that radius is first argument

       @param r Radius (enter a negative radius for an imaginary sphere)
       @param 3 of coordinates


       @sa vsr::nga::dualSph
   */
  static Dls dualSph(VSR_PRECISION r, VSR_PRECISION x, VSR_PRECISION y,
                        VSR_PRECISION z);

  /*! Dual Sph from Coordinate Center (shorthand)


  */
  static Dls dls(VSR_PRECISION r, VSR_PRECISION x, VSR_PRECISION y,
                 VSR_PRECISION z) {
    return dualSph(r, x, y, z);
  }

  /*! Dual Sph from Element FIRST and Radius
      @param v vsr::cga::Vec (function will take first 3 weights)
      @param Radius (enter a negative radius for an imaginary sphere)
  */
  static Dls dls(const Vec& v, VSR_PRECISION r = 1.0);

  /*! Dual Sph from Element FIRST and Radius
      @param Any input Multivector v (function will take first 3 weights)
      @param Radius (enter a negative radius for an imaginary sphere)
  */
  static Dls sphere(const Pnt& v, VSR_PRECISION r = 1.0);

  /*! Dual Sph from Pnt and Radius (faster)
      @param Pnt
      @param Radius (enter a negative radius for an imaginary sphere)
  */
  static Dls dls(const Pnt& p, VSR_PRECISION r = 1.0);
  ///@}

  ///@{
  /*!
     Simple Center of A Round (not normalized -- use loc or location method
     instead)

     @sa nga::Round::center for @ref generic case
  */
  static Pnt center(const Dls& s);
  /*
   Simple Center of A Par (not normalized -- use loc or location method
   instead)
  */
  static Pnt center(const Par& s);
  /*
   Simple Center of A Cir (not normalized -- use loc or location method
   instead)
  */
  static Pnt center(const Cir& s);
  /*
   Simple Center of A Sph (not normalized -- use loc or location method
   instead)
  */
  static Pnt center(const Sph& s);
  /*!
    Simple Center of a @ref round Element (shorthand)
    @ingroup shorthand

    @sa nga::Round::cen for @ref generic implementation
  */
  template <class A>
  static Pnt cen(const A& s) {
    return center(s);
  }

  /*!
    Location (normalized) of A Round Element (normalized) (Shorthand)

    @sa nga::Round::location for @ref generic implementation
  */
  template <class A>
  static Pnt location(const A& s) {
    return null(cen(s));
  }

  /*!
   Location (normalizd) of a @ref round Element (shorthand)
   @ingroup shorthand

   @sa nga::Round::cen for @ref generic implementation
 */

  template <class A>
  static Pnt loc(const A& s) {
    return location(s);
  }
  ///@}

  ///@{
  /*! Squared Size of a Dls (result could be negative)
      @param input normalized dual sphere

      @sa vsr::nga::size for general case
  */
  static VSR_PRECISION size(const Dls& s, bool bDual = true);

  /*! Squared Size of a Pnt Par (result could be negative)
     @param input normalized vsr::cga::Par

     @sa vsr::nga::size for general case
 */
  static VSR_PRECISION size(const Par& s, bool bDual = true);

  /*! Squared Size of a Cir (result could be negative)
      @param input normalized vsr::cga::Cir

      @sa vsr::nga::size for general case
  */
  static VSR_PRECISION size(const Cir& s, bool bDual = false);

  /*! Squared Size of a Sph (result could be negative)
      @param input normalized vsr::cga::Sph

      @sa vsr::nga::size for general case
  */
  static VSR_PRECISION size(const Sph& s, bool bDual = false);

  /*! Squared Size of Normalized Dual Sph (faster than general case)
     @param Normalized Dual Sph

     @sa vsr::nga::dsize
 */
  static VSR_PRECISION dsize(const Pnt& dls);

  /*! Radius of Dls
  */
  static VSR_PRECISION radius(const Dls& s);

  /*! Radius of Par
  */
  static VSR_PRECISION radius(const Par& s);

  /*! Radius of Cir
  */
  static VSR_PRECISION radius(const Cir& s);

  /*! Radius ofSph
  */
  static VSR_PRECISION radius(const Sph& s);

  template <class T>
  static VSR_PRECISION rad(const T& t) {
    return radius(t);
  }

  /*! Curvature of Round
      @param Round Element
  */
  template <class A>
  static VSR_PRECISION curvature(const A& s) {
    VSR_PRECISION r = rad(s);
    return (r == 0) ? 10000 : 1.0 / rad(s);
  }

  /*! Curvature of Round
      @param Round Element
  */
  template <class T>
  static VSR_PRECISION cur(const T& t) {
    return curvature(t);
  }
  ///@}

  ///@{
  /*! Squared distance between two points

    @sa vsr::nga::squaredDistance
 */
  static VSR_PRECISION squaredDistance(const Pnt& a, const Pnt b);

  /*! Squared distance between two points
  */
  static VSR_PRECISION sqd(const Pnt& a, const Pnt b) {
    return squaredDistance(a, b);
  }

  /*! Distance between points a and b */
  static VSR_PRECISION distance(const Pnt& a, const Pnt b);

  /*! Distance between points a and b (shorthand)*/
  static VSR_PRECISION dist(const Pnt& a, const Pnt& b) {
    return distance(a, b);
  }
  ///@}

  ///@{
  /*! Split Pnts from Pnt Par
      @param PntPar input
      returns a vector<Pnt>
  */
  static std::vector<Pnt> split(const Par& pp);

  /*! Split Pnts from Pnt Par and normalize
      @param PntPar input
      returns a vector<Pnt>
  */
  static std::vector<Pnt> splitLocation(const Par& pp);

  /*!
   * Split a point pair and return one
   * @param pp Pnt Par
   * @param bFirst which one to return
   * */
  static Pnt split(const Par& pp, bool bFirst);

  /*!
   * Split A Cir into its dual point pair poles
  */
  static std::vector<Pnt> split(const Cir& nc) { return split(nc.dual()); }
  ///@}

  ///@{
  /*! Direction of a Par
       @ingroup direction
       @param p cga::Par

       @sa nga::direction for @ref generic implementation
   */
  static Drv direction(const Par& p);

  /*! Direction of a Cir
       @ingroup direction
       @param c cga::Cir

       @sa nga::direction for @ref generic implementation
   */

  static Drb direction(const Cir& c);

  /*! Direction of a Sph
       @param c cga::Sph

       @sa nga::direction for @ref generic implementation
   */

  static Drb direction(const Sph& c);

  /*! Direction of Round Element (shorthand)
      @param s a Direct @ref round
  */
  template <class A>
  static auto dir(const A& s) -> decltype(direction(s)) {
    return direction(s);
  }
  ///@}

  ///@{
  /*! Carrier Flat of Par
       @param p vsr::cga::Par

       @sa nga::Round::carrier for @ref generic implementation
   * */
  static Lin carrier(const Par& p);

  /*! Carrier Flat of Cir
       @param c vsr::cga::Cir

       @sa nga::Round::carrier for @ref generic implementation
   * */

  static Pln carrier(const Cir& c);

  /*! Carrier Flat of Direct? Round Element (Shorthand)
      @ingroup shorthand

      @sa nga::Round::car for @ref generic implementation

  */
  template <class A>
  static auto car(const A& s) -> decltype(carrier(s)) {
    return carrier(s);
  }
  ///@}

  /*! Dual Surround of a Direct or Dual Par
      @sa nga::Round::surround for @ref generic implementation
  */
  static Dls surround(const Par& s);

  /*! Dual Surround of a Direct or Dual Cir
      @sa nga::Round::surround for @ref generic implementation
  */
  static Dls surround(const Cir& s);

  /*! Dual Surround of a Direct or Dual Round Element (Shorthand)
      @ingroup shorthand

      @sa nga::Round::sur for @ref generic implementation
   */
  template <class A>
  static Dls sur(const A& s) {
    return surround(s);
  }

  /*!
   Par From Dls and @ref euclidean subspace Bivector
   Note: Result will be imaginary if input Dls is real . .

     @param dls Dls
     @param v Vector

     @returns Par

     @sa nga::Round::produce for @ref generic implementation

   */
  static Par produce(const Dls& dls, const Vec& v);

  /*!
   Cir From Dls and @ref euclidean subspace Bivector
   Note: Result will be imaginary if input Dls is real . .

     @param dls Dls
     @param b Bivector

     @returns Cir

     @sa nga::Round::produce for @ref generic implementation

   */
  static Cir produce(const Dls& dls, const Biv& b);

  //    /*
  //      Creates a real / imaginary round from an imaginary / real round
  //     */
  //     template<class A>
  //     auto
  //     real(const A& s) RETURNS (
  //         produce(
  //                Round::dls( Round::loc( s ), -Round::rad( Round::sur( s ) )
  //                ),
  //                typename A::space::origin(-1) <= Round::dir( s )
  //              )
  //     )
  //
  //
  //    /*
  //      Creates an imaginary round from an real round
  //     */
  //     template<class A>
  //     auto
  //     imag(const A& s) RETURNS (
  //         produce(
  //                Round::dls( Round::loc( s ), Round::rad( Round::sur( s ) )
  //                ),
  //                typename A::space::origin(-1) <= Round::dir( s )
  //              )
  //     )

  /*!
    Dual Round from Center and Pnt on Surface
     @param c Pnt at center
     @param p Pnt on surface
   * */
  static Dls at(const Pnt& c, const Pnt& p);

  /*!
     Direct Pnt From Dual Sph and Euclidean Carrier Flat
     @sa vsr::nga::pnt for the @ref generic implementation
   */
  static Pnt point(const Dls& dls, const Vec& flat);

  /*! Euclidean Vector of Cir at theta

      @sa vsr::nga::vec for the @ref generic implementation
  */
  static Vec vec(const Cir& c, VSR_PRECISION theta = 0);

  /*! Pnt Par on Direct Cir at angle t
      @sa vsr::nga::par_cir for the @ref generic implementation
  */
  static Par pair(const Cir& c, VSR_PRECISION t);

  /*! Pnt on Cir at angle t
      @sa vsr::nga::pnt_cir for the @ref generic implementation
  */
  static Pnt point(const Cir& c, VSR_PRECISION t);
};

/*!
 *  3D operations on @ref flat types
    @ingroup cgaops

   @sa @ref flat for a list of @ref cgatypes on which these methods operate
   @sa vsr::nga::Flat for  @ref generic n-dimensional implementation details
 *
 */
struct Flat {
  ///@{
  /*! Direction of Lin

        @param f Lin
        @returns a @ref direction
    */
  static Drv direction(const Lin& f);

  /*! Direction of Pln

       @param f Pln
       @returns a @ref direction
   */
  static Drb direction(const Pln& f);

  /*! direction shorthand
      @ingroup shorthand

      @param s Lin or Pln
      @returns a @ref direction

      @sa nga::Flat::dir for @ref generic implementation
  */
  template <class A>
  static auto dir(const A& s) -> decltype(direction(s)) {
    return direction(s);
  }
  ///@}

  ///@{
  /*! Location of Dll closest to Pnt p

        @param f Dll
        @param p Pnt

        @returns Pnt

        @sa nga::Flat::location for the @ref generic implementation
    */
  static Pnt location(const Dll& f, const Pnt& p, bool dual = true);

  /*! Location of Lin closest to Pnt p

        @param f Lin
        @param p Pnt

        @returns conformal point in same metric as f

        @sa nga::Flat::location for the @ref generic implementation
    */
  static Pnt location(const Lin& f, const Pnt& p, bool dual = false);

  /*! Location of Dlp closest to Pnt p

        @param f Dlp
        @param p Pnt

        @returns Pnt

        @sa nga::Flat::location for the @ref generic implementation
    */
  static Pnt location(const Dlp& f, const Pnt& p, bool dual = true);

  /*! Location of Pln closest to Pnt p

        @param f Pln
        @param p Pnt

        @returns Pnt

        @sa nga::Flat::location for the @ref generic implementation
    */
  static Pnt location(const Pln& f, const Pnt& p, bool dual = false);

  /*! Location of flat (shorthand)
      @ingroup shorthand

      three-letter version of cga::Flat::location

      @sa nga::Flat::loc for the @ref generic implementation
  */
  template <class A>
  static Pnt loc(const A& a, const Pnt& p, bool dual) {
    return location(a, p);
  }
  ///@}

  /*! Weight of Dll

     @param f Dual or Direct Flat type e.g. vsr::cga::Lin or vsr::cga::Dll
     @param bDual boolean flag for whether first argument is a dual
 */
  static VSR_PRECISION wt(const Dll& f, bool bDual = true);

  /*! Dual Pln from Pnt and Direction */
  static Dlp plane(const Pnt& pnt, const Drv& drv);

  /*! Direct Lin at origin with coordinate v ... */
  static Lin line(VSR_PRECISION x, VSR_PRECISION y, VSR_PRECISION z);

  /*! Direct hyperbolic d-Lin at origin with coordinate v ... */
  static Cir dline(VSR_PRECISION x, VSR_PRECISION y, VSR_PRECISION z);
};

/*!
 *  3D operations on @ref tangent types
    @ingroup cgaops

    @sa @ref tangent for a list of @ref cgatypes on which these methods operate
    @sa vsr::nga::Tangent for the @ref generic  n-dimensional implementation
 details

 */
struct Tangent {
  /*! Tangent Element of A Cir at Pnt p

      @param r Cir
      @param p Pnt

      @return a Par @ref Tangent
  */
  static Par at(const Cir& r, const Pnt& p);

  /*! Tangent Element of A Sph at Pnt p

      @param r Sph
      @param p Pnt

      @return a Cir @ref Tangent
  */
  static Cir at(const Sph& r, const Pnt& p);

  /*! Weight of Tnv
      @ingroup euclidean
   */
  static VSR_PRECISION wt(const Tnv& s);

  /*! Weight of TangentBiVector
     @ingroup euclidean
  */
  static VSR_PRECISION wt(const Tnb& s);

  /*! Weight of Tnt
      @ingroup euclidean
   */
  static VSR_PRECISION wt(const Tnt& s);
};
}
}  // vsr::cga::

#endif /* ----- #ifndef vsr_cga3D_round_INC  ----- */
