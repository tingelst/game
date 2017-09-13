#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_PLUGIN(linear_solver) {
  py::module m("linear_solver", "linear_solver");

  m.def("solve",
        [](Eigen::Ref<Eigen::MatrixXd> H, Eigen::Ref<Eigen::VectorXd> g) {
          Eigen::MatrixXd solution = H.fullPivHouseholderQr().solve(g);
          return solution;
        });

  return m.ptr();
}
