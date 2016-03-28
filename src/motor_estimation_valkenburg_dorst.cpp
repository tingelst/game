#include <game/vsr/cga_op.h>
#include <game/vsr/generic_op.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

using namespace vsr::cga;
namespace py = pybind11;

namespace {
template <typename ScalarType, int Rows, int Cols>
py::array_t<ScalarType> Matrix() {
  return py::array(py::buffer_info(
      nullptr, /* Pointer to data (nullptr -> ask NumPy to allocate!) */
      sizeof(ScalarType),                         /* Size of one item */
      py::format_descriptor<ScalarType>::value(), /* Buffer format */
      2,                                          /* How many dimensions? */
      {Rows, Cols}, /* Number of elements for each dimension */
      {sizeof(double) * Rows, sizeof(double)} /* Strides for each dimension */
      ));
}
}
PYBIND11_PLUGIN(motor_estimation_valkenburg_dorst) {
  py::module m("motor_estimation_valkenburg_dorst",
               "motor_estimation_valkenburg_dorst");
  m.def("lx", [](const Pnt &P, const Pnt &Q, const Rot &X) {
    auto L = (Q * X * P + ~Q * X * ~P) / X;
    std::cout << L << std::endl;
  });

  return m.ptr();
}
