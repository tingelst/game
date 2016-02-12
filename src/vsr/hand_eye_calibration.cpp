#include <pybind11/pybind11.h>

#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>

#include "game/vsr/vsr.h"
#include "game/motor_parameterization.h"

namespace py = pybind11;

using vsr::cga::Scalar;
using vsr::cga::Vector;
using vsr::cga::Bivector;
using vsr::cga::Point;
using vsr::cga::DualLine;
using vsr::cga::Motor;
using vsr::cga::Rotor;
using vsr::cga::Translator;
using vsr::cga::Origin;
using vsr::cga::Infinity;
using vsr::cga::Pnt;
using vsr::cga::Dll;
using vsr::cga::Mot;
using vsr::nga::Op;

namespace game {

class HandEyeCalibrator {
 public:
  MotorEstimationSolver() {}
  MotorEstimationSolver(const HandEyeCalibrator &hand_eye_calibrator) {}
  MotorEstimationSolver(const Mot &X, const Mot &Z,
                        const std::vector<double> &intrinsics)
      : X_(X), Z_(Z), intrinsics_(intrinsics) {}

  auto SetMotorParameterizationTypeFromString(const std::string &type) -> void {
    if (type == "POLAR_DECOMPOSITION") {
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

  auto Solve() -> std::tuple<Mot, Mot, std::vector<double>, py::dict> {
    ceres::Solve(options_, &problem_, &summary_);
    return std::make_tuple(motor_, Summary());
  }

  // private:
  Mot X_;
  Mot Z_;
  std::vector<double> intrinsics_;

  ceres::Problem problem_;
  ceres::Solver::Options options_;
  ceres::Solver::Summary summary_;
};

PYBIND11_PLUGIN(hand_eye_calibration) {
  py::module m("hand_eye_calibration", "hand_eye_calibration");

  py::class_<HandEyeCalibrator>(m, "HandEyeCalibrator")
      .def(py::init<const Mot &, const Mot &, const std::vector<double> &>())
      .def("set_parameterization",
           &HandEyeCalibrator::SetMotorParameterizationTypeFromString)
      .def("solve", &HandEyeCalibrator::Solve)
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

}  // namespace game
