#include <pybind11/pybind11.h>

#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>

#include "game/vsr/vsr.h"
#include "game/motor_parameterization.h"
#include "game/ceres_python_utils.h"

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
  HandEyeCalibrator() {}
  HandEyeCalibrator(const HandEyeCalibrator &hand_eye_calibrator) {}
  HandEyeCalibrator(const Mot &X, const Mot &Z,
                    const std::vector<double> &intrinsics)
      : X_(X), Z_(Z), intrinsics_(intrinsics) {}

  struct HandEyeReprojectionError {
    HandEyeReprojectionError(const Mot &A, const Pnt &point, const Vector<double> &pixel)
        : A_(A), point_(point), pixel_(pixel) {}

    template <typename T>
    auto operator()(const T *const X_arr, const T *const Z_arr,
                    const T *const intrinsics_arr, T *residual) const -> bool {
      return true;
    }

   private:
    Mot A_;
    Pnt point_;
    Vector<double> pixel_;
  };

  auto SetMotorParameterizationTypeFromString(const std::string &type) -> void {
    if (type == "POLAR_DECOMPOSITION") {
      std::cout << "game:: Using polar decomposition." << std::endl;
      problem_.SetParameterization(
          &X_[0],
          new ceres::AutoDiffLocalParameterization<MotorPolarDecomposition, 8,
                                                   8>);
      problem_.SetParameterization(
          &Z_[0],
          new ceres::AutoDiffLocalParameterization<MotorPolarDecomposition, 8,
                                                   8>);

    } else if (type == "BIVECTOR_GENERATOR") {
      std::cout << "game:: Using bivector generator (Versor)." << std::endl;
      problem_.SetParameterization(
          &X_[0],
          new ceres::AutoDiffLocalParameterization<MotorFromBivectorGenerator,
                                                   8, 6>);
      problem_.SetParameterization(
          &Z_[0],
          new ceres::AutoDiffLocalParameterization<MotorFromBivectorGenerator,
                                                   8, 6>);
    } else {
      std::cout << "Unknown motor parameterization type" << std::endl;
    }
  }

  auto Summary() const -> py::dict { return game::SummaryToDict(summary_); }

  auto Solve() -> std::tuple<Mot, Mot, std::vector<double>, py::dict> {
    ceres::Solve(options_, &problem_, &summary_);
    return std::make_tuple(X_, Z_, intrinsics_, Summary());
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
                    [](HandEyeCalibrator &instance) {
                      return instance.options_.num_threads;
                    },
                    [](HandEyeCalibrator &instance, int arg) {
                      instance.options_.num_threads = arg;
                    })
      .def_property("num_linear_solve_threads",
                    [](HandEyeCalibrator &instance) {
                      return instance.options_.num_linear_solver_threads;
                    },
                    [](HandEyeCalibrator &instance, int arg) {
                      instance.options_.num_linear_solver_threads = arg;
                    })
      .def_property("max_num_iterations",
                    [](HandEyeCalibrator &instance) {
                      return instance.options_.max_num_iterations;
                    },
                    [](HandEyeCalibrator &instance, int arg) {
                      instance.options_.max_num_iterations = arg;
                    })

      .def_property("parameter_tolerance",
                    [](HandEyeCalibrator &instance) {
                      return instance.options_.parameter_tolerance;
                    },
                    [](HandEyeCalibrator &instance, double arg) {
                      instance.options_.parameter_tolerance = arg;
                    })
      .def_property("function_tolerance",
                    [](HandEyeCalibrator &instance) {
                      return instance.options_.function_tolerance;
                    },
                    [](HandEyeCalibrator &instance, double arg) {
                      instance.options_.function_tolerance = arg;
                    })
      .def_property("gradient_tolerance",
                    [](HandEyeCalibrator &instance) {
                      return instance.options_.gradient_tolerance;
                    },
                    [](HandEyeCalibrator &instance, double arg) {
                      instance.options_.gradient_tolerance = arg;
                    })
      .def_property("minimizer_progress_to_stdout",
                    [](HandEyeCalibrator &instance) {
                      return instance.options_.minimizer_progress_to_stdout;
                    },
                    [](HandEyeCalibrator &instance, double arg) {
                      instance.options_.minimizer_progress_to_stdout = arg;
                    })
      .def_property("minimizer_type",
                    [](HandEyeCalibrator &instance) {
                      return ceres::MinimizerTypeToString(
                          instance.options_.minimizer_type);
                    },
                    [](HandEyeCalibrator &instance, const std::string &arg) {
                      ceres::StringToMinimizerType(
                          arg, &instance.options_.minimizer_type);
                    })
      .def_property("linear_solver_type",
                    [](HandEyeCalibrator &instance) {
                      return ceres::LinearSolverTypeToString(
                          instance.options_.linear_solver_type);
                    },
                    [](HandEyeCalibrator &instance, const std::string &arg) {
                      ceres::StringToLinearSolverType(
                          arg, &instance.options_.linear_solver_type);
                    })
      // .def_property(
      //     "trust_region_minimizer_iterations_to_dump",
      //     [](HandEyeCalibrator &instance) {
      //       return
      //       instance.options_.trust_region_minimizer_iterations_to_dump;
      //     },
      //     [](HandEyeCalibrator &instance, const std::vector<int>& arg) {
      //       std::copy(arg.begin(), arg.end(),
      //                 instance.options_
      //                     .trust_region_minimizer_iterations_to_dump.begin());
      //     })

      ;

  return m.ptr();
}

}  // namespace game
