#include "game/vsr/vsr.h"
#include <game/vsr/cga_op.h>
#include <game/vsr/generic_op.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <iostream>

using vsr::cga::Gen;
using vsr::cga::Vec;
using vsr::cga::Biv;
using vsr::cga::Pnt;
using vsr::cga::Rot;
using vsr::cga::Mot;
using vsr::cga::Trs;
using vsr::cga::Lin;


using namespace vsr::cga;
namespace py = pybind11;

namespace {
template <typename ScalarType, int Rows, int Cols>
py::array_t<ScalarType> Matrix() {
  return py::array(py::buffer_info(
      nullptr, /* Pointer to data (nullptr -> ask NumPy to allocate!) */
      sizeof(ScalarType),                         /* Size of one item */
      py::format_descriptor<ScalarType>::format(), /* Buffer format */
      2,                                          /* How many dimensions? */
      {Rows, Cols}, /* Number of elements for each dimension */
      {sizeof(ScalarType) * Rows, sizeof(ScalarType)}
      /* Strides for each dimension */
      ));
}
}

template <typename MultivectorT>
py::array_t<double> VDMatrix(const MultivectorT &p, const MultivectorT &q) {

  using MotRec = vsr::Multivector<vsr::algebra<vsr::metric<4, 1, true>, double>,
                                  vsr::Basis<0, 3, 5, 6, 9, 10, 12, 15>>;
  std::vector<MotRec> rs;
  rs.reserve(8);

  for (int i = 0; i < 8; ++i) {
    Mot ei;
    ei[i] = 1.0;
    rs[i] = MotRec(~q * ei * p);
  }

  auto matrix = ::Matrix<double, 8, 8>();
  auto buf = matrix.request();
  Eigen::Map<Eigen::Matrix<double, 8, 8>> L(
      reinterpret_cast<double *>(buf.ptr));
  L = Eigen::Matrix<double, 8, 8>::Zero();

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      Mot ei;
      ei[i] = 1.0;
      L(i, j) = (~ei * rs[j])[0];
    }
  }

  return matrix;
}

PYBIND11_PLUGIN(motor_estimation_valkenburg_dorst) {
  py::module m("motor_estimation_valkenburg_dorst",
               "motor_estimation_valkenburg_dorst");
  m.def("point_matrix", &VDMatrix<Pnt>);
  m.def("dual_line_matrix", &VDMatrix<Dll>);
  m.def("dual_plane_matrix", &VDMatrix<Dlp>);
  return m.ptr();
}
