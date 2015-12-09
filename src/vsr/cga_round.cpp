/*
 * =====================================================================================
 *
 *       Filename:  vsr_cga3D_round.cpp
 *
 *    Description:  template specializations of 3D cga round methods
 *
 *        Version:  1.0
 *        Created:  07/13/2015 19:25:38
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Pablo Colapinto (), gmail->wolftype
 *   Organization:  wolftype
 *
 * =====================================================================================
 */

#include "game/vsr/cga_round.h"

namespace vsr {
namespace cga {

/*! Null Pnt from a vec
 */
Pnt Round::null(const Vec& v) { return vsr::nga::Round::null(v); }

/*! Null Pnt from a Pnt
 */
Pnt Round::null(const Pnt& p) { return vsr::nga::Round::null(p); }

/*! Or Null Pnt from Coordinates (x,y,z,...)
 */
Pnt Round::null(VSR_PRECISION x, VSR_PRECISION y, VSR_PRECISION z) {
  return vsr::nga::Round::null(x, y, z);
}

/*! Null Pnt from Coordinates
*/
Pnt Round::point(VSR_PRECISION x, VSR_PRECISION y, VSR_PRECISION z) {
  return vsr::nga::Round::null(x, y, z);
}

/*! Dual Sph from Coordinate Center
     @param 3 of coordinates
     @param Radius (enter a negative radius for an imaginary sphere)
 */
Dls Round::dualSph(VSR_PRECISION r, VSR_PRECISION x, VSR_PRECISION y,
                             VSR_PRECISION z) {
  return vsr::nga::Round::dls(r, x, y, z);
}

/*! Dual Sph from Element FIRST and Radius
    @param v vsr::cga::Vec (function will take first 3 weights)
    @param Radius (enter a negative radius for an imaginary sphere)
*/
Dls Round::dls(const Vec& v, VSR_PRECISION r) {
  return vsr::nga::Round::dls(v, r);
}

/*! Dual Sph from Element FIRST and Radius
    @param Any input Multivector v (function will take first 3 weights)
    @param Radius (enter a negative radius for an imaginary sphere)
*/
Dls Round::sphere(const Pnt& v, VSR_PRECISION r) {
  return vsr::nga::Round::sphere(v, r);
}

/*! Dual Sph from Pnt and Radius (faster)
    @param Pnt
    @param Radius (enter a negative radius for an imaginary sphere)
*/
Dls Round::dls(const Pnt& p, VSR_PRECISION r) {
  return vsr::nga::Round::dls(p, r);
}

/*!
 Simple Center of A Dls (not normalized -- use loc or location method)
*/
Pnt Round::center(const Dls& s) { return vsr::nga::Round::center(s); }
/*!
 Simple Center of A Par (not normalized -- use loc or location method)
*/
Pnt Round::center(const Par& s) { return vsr::nga::Round::center(s); }
/*!
 Simple Center of A Cir (not normalized -- use loc or location method)
*/
Pnt Round::center(const Cir& s) { return vsr::nga::Round::center(s); }
/*!
 Simple Center of A Sph (not normalized -- use loc or location method)
*/
Pnt Round::center(const Sph& s) { return vsr::nga::Round::center(s); }

/*! Squared Size of a Dls (result could be negative)
    @param input normalized dual sphere

    @sa vsr::nga::size for general case
*/
VSR_PRECISION Round::size(const Dls& s, bool bDual) {
  return vsr::nga::Round::size(s, true);
}

/*! Squared Size of a Pnt Par (result could be negative)
   @param input normalized vsr::cga::Par

   @sa vsr::nga::size for general case
*/
VSR_PRECISION Round::size(const Par& s, bool bDual) {
  return vsr::nga::Round::size(s, bDual);
}

/*! Squared Size of a Cir (result could be negative)
    @param input normalized vsr::cga::Cir

    @sa vsr::nga::size for general case
*/
VSR_PRECISION Round::size(const Cir& s, bool bDual) {
  return vsr::nga::Round::size(s, bDual);
}

/*! Squared Size of a Sph (result could be negative)
    @param input normalized vsr::cga::Sph

    @sa vsr::nga::size for general case
*/
VSR_PRECISION Round::size(const Sph& s, bool bDual) {
  return vsr::nga::Round::size(s, false);
}

/*! Radius of Dls
    @ingroup euclidean
*/
VSR_PRECISION Round::radius(const Dls& s) {
  return vsr::nga::Round::radius(s);
}

/*! Radius of Par
    @ingroup euclidean
*/
VSR_PRECISION Round::radius(const Par& s) {
  return vsr::nga::Round::radius(s);
}

/*! Radius of Cir
    @ingroup euclidean
*/
VSR_PRECISION Round::radius(const Cir& s) {
  return vsr::nga::Round::radius(s);
}

/*! Radius ofSph
    @ingroup euclidean
*/
VSR_PRECISION Round::radius(const Sph& s) {
  return vsr::nga::Round::radius(s);
}

/*! Squared Size of Normalized Dual Sph (faster than general case)
    @ingroup euclidean
    @param Normalized Dual Sph

    @sa vsr::nga::dsize
*/
VSR_PRECISION Round::dsize(const Pnt& dls) {
  return vsr::nga::Round::dsize(dls);
}

/*! Squared distance between two points
  @ingroup euclidean

  @sa vsr::nga::squaredDistance
*/
VSR_PRECISION Round::squaredDistance(const Pnt& a, const Pnt b) {
  return vsr::nga::Round::squaredDistance(a, b);
}

/*! Distance between points a and b */
VSR_PRECISION Round::distance(const Pnt& a, const Pnt b) {
  return vsr::nga::Round::distance(a, b);
}

/*! Split Pnts from Pnt Par
    @param PntPar input
    returns a vector<Pnt>
*/
std::vector<Pnt> Round::split(const Par& pp) {
  return vsr::nga::Round::split(pp);
}

/*! Split Pnts from Pnt Par and normalize
    @param PntPar input
    returns a vector<Pnt>
*/
std::vector<Pnt> Round::splitLocation(const Par& pp) {
  return vsr::nga::Round::splitLocation(pp);
}

/*!
 * Split a point pair and return one
 * @param pp Pnt Par
 * @param bFirst which one to return
 * */
Pnt Round::split(const Par& pp, bool bFirst) {
  return vsr::nga::Round::split(pp, bFirst);
}

/*! Direction of a Par
     @ingroup direction
     @param p vsr::cga::Par

     @sa vsr::nga::direction
 */
Drv Round::direction(const Par& p) {
  return vsr::nga::Round::direction(p);
}

/*! Direction of a Cir
     @ingroup direction
     @param c vsr::cga::Cir

     @sa vsr::nga::direction
 */

Drb Round::direction(const Cir& c) {
  return vsr::nga::Round::direction(c);
}

/*! Direction of a Sph
     @ingroup direction
     @param c vsr::cga::Sph

     @sa vsr::nga::direction
 */

Drb Round::direction(const Sph& c) {
  return vsr::nga::Round::direction(c);
}

/*! Carrier Flat of Par
     @ingroup flat
     @param p vsr::cga::Par

     @sa vsr::nga::carrier
 * */
Lin Round::carrier(const Par& p) { return vsr::nga::Round::carrier(p); }

/*! Carrier Flat of Cir
     @ingroup flat
     @param c vsr::cga::Cir

     @sa vsr::nga::carrier
 * */

Pln Round::carrier(const Cir& c) { return vsr::nga::Round::carrier(c); }

/*! Dual Surround of a Direct or Dual Par
    @sa vsr::nga::surround for the general case
*/
Dls Round::surround(const Par& s) {
  return vsr::nga::Round::surround(s);
}

/*! Dual Surround of a Direct or Dual Cir
    @sa vsr::nga::surround for the general case
*/
Dls Round::surround(const Cir& s) {
  return vsr::nga::Round::surround(s);
}

/*!
 Direct Round From Dual Sph and Euclidean Bivector
 Note: round will be imaginary if dual sphere is real . .

   @sa vsr::nga::Round

 */
Par Round::produce(const Dls& dls, const Vec& v) {
  return vsr::nga::Round::produce(dls, v);
}

/*!
 Direct Round From Dual Sph and Euclidean Bivector
 Note: round will be imaginary if dual sphere is real . .

   @sa vsr::nga::Round

 */
Cir Round::produce(const Dls& dls, const Biv& v) {
  return vsr::nga::Round::produce(dls, v);
}

//    /*!
//      Creates a real / imaginary round from an imaginary / real round
//     */
//     template<class A>
//     auto
//     real(const A& s) RETURNS (
//         produce(
//                Round::dls( Round::loc( s ), -Round::rad( Round::sur( s ) ) ),
//                typename A::space::origin(-1) <= Round::dir( s )
//              )
//     )
//
//
//    /*!
//      Creates an imaginary round from an real round
//     */
//     template<class A>
//     auto
//     imag(const A& s) RETURNS (
//         produce(
//                Round::dls( Round::loc( s ), Round::rad( Round::sur( s ) ) ),
//                typename A::space::origin(-1) <= Round::dir( s )
//              )
//     )

/*!
  Dual Round from Center and Pnt on Surface
   @param Center
   @param point on surface
 * */
Dls Round::at(const Dls& c, const Dls& p) {
  return vsr::nga::Round::at(c, p);
}

/*!
   Direct Pnt From Dual Sph and Euclidean Carrier Flat
   @sa vsr::nga::pnt
 */
Pnt Round::point(const Dls& dls, const Vec& flat) {
  return vsr::nga::Round::pnt(dls, flat);
}

/*! Euclidean Vector of Cir at theta
    @include euclidean

    @sa vsr::nga::vec
*/
Vec Round::vec(const Cir& c, VSR_PRECISION theta) {
  return vsr::nga::Round::vec(c, theta);
}

/*! Pnt Par on Direct Cir at angle t
    @sa vsr::nga::par_cir
*/
Par Round::pair(const Cir& c, VSR_PRECISION t) {
  return vsr::nga::Round::par_cir(c, t);
}

/*! Pnt on Cir at angle t
    @sa vsr::nga::pnt_cir
*/
Pnt Round::point(const Cir& c, VSR_PRECISION t) {
  return vsr::nga::Round::pnt_cir(c, t);
}

/*! Direction of Lin
      @ingroup direction

      @param f Lin
      @returns a @ref direction
  */
Drv Flat::direction(const Lin& f) {
  return nga::Flat::direction(f);
}

/*! Direction of Pln
     @ingroup direction

     @param f Pln
     @returns a @ref direction
 */
Drb Flat::direction(const Pln& f) {
  return nga::Flat::direction(f);
}

/*! Location of Dll closest to Pnt p
      @ingroup round

      @param f Dll
      @param p Pnt

      @returns Pnt

      @sa vsr::nga::Flat::location for the generic ND case
  */
Pnt Flat::location(const Dll& f, const Pnt& p, bool dual) {
  return nga::Flat::location(f, p, dual);
}

/*! Location of Lin closest to Pnt p
      @ingroup round

      @param f Lin
      @param p Pnt

      @returns conformal point in same metric as f

      @sa vsr::nga::Flat::location for the generic ND case
  */
Pnt Flat::location(const Lin& f, const Pnt& p, bool dual) {
  return nga::Flat::location(f, p, dual);
}

/*! Location of Dlp closest to Pnt p
      @ingroup round

      @param f Dlp
      @param p Pnt

      @returns Pnt

      @sa vsr::nga::Flat::location for the generic ND case
  */
Pnt Flat::location(const Dlp& f, const Pnt& p, bool dual) {
  return nga::Flat::location(f, p, dual);
}

/*! Location of Pln closest to Pnt p
      @ingroup round

      @param f Dll
      @param p Pnt

      @returns Pnt

      @sa vsr::nga::Flat::location for the generic ND case
  */
Pnt Flat::location(const Pln& f, const Pnt& p, bool dual) {
  return nga::Flat::location(f, p, dual);
}

/*! Weight of Dll
     @ingroup euclidean

     @param f Dual or Direct Flat type e.g. vsr::cga::Lin or vsr::cga::Dll
     @param bDual boolean flag for whether first argument is a dual
 */
VSR_PRECISION Flat::wt(const Dll& f, bool bDual) {
  return nga::Flat::wt(f, bDual);
}

/*! Dual Pln from Pnt and Direction */
Dlp Flat::plane(const Pnt& pnt, const Drv& drv) {
  return nga::Flat::dlp(pnt, drv);
}

/*! Direct Lin at origin with coordinate v ... */
Lin Flat::line(VSR_PRECISION x, VSR_PRECISION y, VSR_PRECISION z) {
  return nga::Flat::line(x, y, z);
}

/*! Direct hyperbolic d-Lin at origin with coordinate v ... */
Cir Flat::dline(VSR_PRECISION x, VSR_PRECISION y, VSR_PRECISION z) {
  return nga::Flat::dline(x, y, z);
}

/*! Tangent Element of A Cir at Pnt p
   @ingroup tangent

   @param r Cir
   @param p Pnt

   @return a Par @ref Tangent
*/
Par Tangent::at(const Cir& r, const Pnt& p) {
  return nga::Tangent::at(r, p);
}

/*! Tangent Element of A Sph at Pnt p
    @ingroup tangent

    @param r Sph
    @param p Pnt

    @return a Cir @ref Tangent
*/
Cir Tangent::at(const Sph& r, const Pnt& p) {
  return nga::Tangent::at(r, p);
}

/*! Weight of Tnv
    @ingroup euclidean
 */
VSR_PRECISION Tangent::wt(const Tnv& s) {
  return nga::Tangent::wt(s);
}

/*! Weight of TangentBiVector
   @ingroup euclidean
*/
VSR_PRECISION Tangent::wt(const Tnb& s) {
  return nga::Tangent::wt(s);
}

/*! Weight of Tnt
    @ingroup euclidean
 */
VSR_PRECISION Tangent::wt(const Tnt& s) {
  return nga::Tangent::wt(s);
}
}
}  // vsr::cga::
