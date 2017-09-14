#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
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
  m.def("solve",
        [](Eigen::Ref<Eigen::MatrixXd> H, Eigen::Ref<Eigen::MatrixXd> g) {
          Eigen::MatrixXd solution = H.fullPivHouseholderQr().solve(g);
          return solution;
        });

  m.def("expm", [](Eigen::Ref<Eigen::MatrixXd> A) {
    Eigen::MatrixXd solution = A.exp();
    return solution;

  });

  m.def("logm", [](Eigen::Ref<Eigen::MatrixXd> A) {
      Eigen::MatrixXd solution = A.log();
      return solution;

    });

  return m.ptr();
}
