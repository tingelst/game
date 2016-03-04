//
// Created by lars on 17.11.15.
//

#include <pybind11/pybind11.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>
#include <iostream>

#include "game/vsr/vsr.h"
#include "game/ceres_python_utils.h"
#include "game/motor_parameterization.h"

#include <vahlen/vahlen.h>

namespace py = pybind11;

using vsr::cga::Scalar;
using vsr::cga::Vector;
using vsr::cga::Bivector;
using vsr::cga::Point;
using vsr::cga::DualLine;
using vsr::cga::DualPlane;
using vsr::cga::Motor;
using vsr::cga::Rotor;
using vsr::cga::Translator;
using vsr::cga::Origin;
using vsr::cga::Infinity;
using vsr::cga::TangentVector;
using vsr::cga::DirectionVector;
using vsr::cga::Pnt;
using vsr::cga::Dll;
using vsr::cga::Dlp;
using vsr::cga::Mot;
using vsr::cga::Tnv;
using vsr::nga::Op;

namespace game {

class MotorEstimationSolver {
public:
  MotorEstimationSolver() {}
  MotorEstimationSolver(const MotorEstimationSolver &motor_estimation_solver) {}
  MotorEstimationSolver(const Mot &motor) : motor_(motor) {
    // options_.trust_region_minimizer_iterations_to_dump =
    //     std::vector<int>{0, 1, 2, 3, 4};
    // options_.trust_region_problem_dump_directory =
    //     std::string{"/Users/lars/devel/game_ws/dump"};
  }

  struct PointCorrespondencesCostFunctor {
    PointCorrespondencesCostFunctor(const Pnt &a, const Pnt &b)
        : a_(a), b_(b) {}

    template <typename T>
    auto operator()(const T *const motor, T *residual) const -> bool {
      Motor<T> M(motor);
      Point<T> a(a_);
      Point<T> b(b_);
      Point<T> c = a.spin(M);

      for (int i = 0; i < 3; ++i) {
        residual[i] = c[i] - b[i];
      }

      return true;
    }

  private:
    const Pnt a_;
    const Pnt b_;
  };

  struct VahlenPointCorrespondencesCostFunctor {
    VahlenPointCorrespondencesCostFunctor(const Pnt &a, const Pnt &b)
        : a_(a), b_(b) {}

    template <typename T>
    auto operator()(const T *const motor, T *residual) const -> bool {
      vahlen::Matrix<T> M = vahlen::Motor<T>(motor);
      vahlen::Matrix<T> a = vahlen::Point<T>(T(a_[0]), T(a_[1]), T(a_[2]));
      vahlen::Matrix<T> b = vahlen::Point<T>(T(b_[0]), T(b_[1]), T(b_[2]));
      vahlen::Matrix<T> c = M * a * vahlen::Reverse(M);

      residual[0] = c(2,0) - b(2,0);
      residual[1] = c(3,0) - b(3,0);
      residual[2] = c(0,0) - b(0,0);

      return true;
    }

  private:
    const Pnt a_;
    const Pnt b_;
  };

  auto Summary() const -> py::dict { return game::SummaryToDict(summary_); }

  auto AddPointCorrespondencesResidualBlock(const Pnt &a, const Pnt &b)
      -> bool {
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<PointCorrespondencesCostFunctor, 3, 8>(
            new PointCorrespondencesCostFunctor(a, b));
    problem_.AddResidualBlock(cost_function, NULL, &motor_[0]);
    return true;
  }

  auto AddVahlenPointCorrespondencesResidualBlock(const Pnt &a, const Pnt &b)
      -> bool {
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<VahlenPointCorrespondencesCostFunctor,
                                        3, 8>(
            new VahlenPointCorrespondencesCostFunctor(a, b));
    problem_.AddResidualBlock(cost_function, NULL, &motor_[0]);
    return true;
  }

  auto SetMotorParameterizationTypeFromString(const std::string &type) -> void {
    if (type == "NORMALIZE") {
      std::cout << "game:: Using rotor normalization." << std::endl;
      problem_.SetParameterization(
          &motor_[0],
          new ceres::AutoDiffLocalParameterization<MotorNormalizeRotor, 8, 8>);
    } else if (type == "POLAR_DECOMPOSITION") {
      std::cout << "game:: Using polar decomposition." << std::endl;
      problem_.SetParameterization(
          &motor_[0],
          new ceres::AutoDiffLocalParameterization<MotorPolarDecomposition, 8,
                                                   8>);
    } else if (type == "BIVECTOR_GENERATOR") {
      std::cout << "game:: Using bivector generator (Versor)." << std::endl;
      problem_.SetParameterization(
          &motor_[0],
          new ceres::AutoDiffLocalParameterization<MotorFromBivectorGenerator,
                                                   8, 6>);
    } else {
      std::cout << "Unknown motor parameterization type" << std::endl;
    }
  }

  auto Solve() -> std::tuple<Mot, py::dict> {
    ceres::Solve(options_, &problem_, &summary_);
    return std::make_tuple(motor_, Summary());
  }

  // private:
  Mot motor_;

  ceres::Problem problem_;
  ceres::Solver::Options options_;
  ceres::Solver::Summary summary_;
};

PYBIND11_PLUGIN(motor_estimation_vahlen) {
  py::module m("motor_estimation_vahlen", "motor estimation vahlen");

  py::class_<MotorEstimationSolver>(m, "MotorEstimationSolver")
      .def(py::init<const Mot &>())
      .def("add_point_correspondences_residual_block",
           &MotorEstimationSolver::AddPointCorrespondencesResidualBlock)
      .def("add_vahlen_point_correspondences_residual_block",
           &MotorEstimationSolver::AddVahlenPointCorrespondencesResidualBlock)
      .def("set_parameterization",
           &MotorEstimationSolver::SetMotorParameterizationTypeFromString)

      .def("solve", &MotorEstimationSolver::Solve)
      .def_property("num_threads",
                    [](MotorEstimationSolver &instance) {
                      return instance.options_.num_threads;
                    },
                    [](MotorEstimationSolver &instance, int arg) {
                      instance.options_.num_threads = arg;
                    })
      .def_property("num_linear_solve_threads",
                    [](MotorEstimationSolver &instance) {
                      return instance.options_.num_linear_solver_threads;
                    },
                    [](MotorEstimationSolver &instance, int arg) {
                      instance.options_.num_linear_solver_threads = arg;
                    })
      .def_property("max_num_iterations",
                    [](MotorEstimationSolver &instance) {
                      return instance.options_.max_num_iterations;
                    },
                    [](MotorEstimationSolver &instance, int arg) {
                      instance.options_.max_num_iterations = arg;
                    })

      .def_property("parameter_tolerance",
                    [](MotorEstimationSolver &instance) {
                      return instance.options_.parameter_tolerance;
                    },
                    [](MotorEstimationSolver &instance, double arg) {
                      instance.options_.parameter_tolerance = arg;
                    })
      .def_property("function_tolerance",
                    [](MotorEstimationSolver &instance) {
                      return instance.options_.function_tolerance;
                    },
                    [](MotorEstimationSolver &instance, double arg) {
                      instance.options_.function_tolerance = arg;
                    })
      .def_property("gradient_tolerance",
                    [](MotorEstimationSolver &instance) {
                      return instance.options_.gradient_tolerance;
                    },
                    [](MotorEstimationSolver &instance, double arg) {
                      instance.options_.gradient_tolerance = arg;
                    })
      .def_property("minimizer_progress_to_stdout",
                    [](MotorEstimationSolver &instance) {
                      return instance.options_.minimizer_progress_to_stdout;
                    },
                    [](MotorEstimationSolver &instance, double arg) {
                      instance.options_.minimizer_progress_to_stdout = arg;
                    })
      .def_property(
          "minimizer_type",
          [](MotorEstimationSolver &instance) {
            return ceres::MinimizerTypeToString(
                instance.options_.minimizer_type);
          },
          [](MotorEstimationSolver &instance, const std::string &arg) {
            ceres::StringToMinimizerType(arg,
                                         &instance.options_.minimizer_type);
          })
      .def_property(
          "linear_solver_type",
          [](MotorEstimationSolver &instance) {
            return ceres::LinearSolverTypeToString(
                instance.options_.linear_solver_type);
          },
          [](MotorEstimationSolver &instance, const std::string &arg) {
            ceres::StringToLinearSolverType(
                arg, &instance.options_.linear_solver_type);
          })
      // .def_property(
      //     "trust_region_minimizer_iterations_to_dump",
      //     [](MotorEstimationSolver &instance) {
      //       return
      //       instance.options_.trust_region_minimizer_iterations_to_dump;
      //     },
      //     [](MotorEstimationSolver &instance, const std::vector<int>& arg) {
      //       std::copy(arg.begin(), arg.end(),
      //                 instance.options_
      //                     .trust_region_minimizer_iterations_to_dump.begin());
      //     })

      ;

  return m.ptr();
}

} // namespace game
