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

  // The intrinsics need to get combined into a single parameter block; use
  // these
  // enums to index instead of numeric constants.
  enum {
    OFFSET_FOCAL_LENGTH_X,
    OFFSET_FOCAL_LENGTH_Y,
    OFFSET_PRINCIPAL_POINT_X,
    OFFSET_PRINCIPAL_POINT_Y,
    OFFSET_K1,
    OFFSET_K2,
    OFFSET_P1,
    OFFSET_P2,
    OFFSET_K3,
  };

  struct AXZBError {
    AXZBError(const Mot &A, const Mot &B, const Pnt &p) : A_(A), B_(B), p_(p) {}

    template <typename T>
    auto operator()(const T *const X_arr, const T *const Z_arr,
                    T *residuals) const -> bool {
      Motor<T> A(A_);
      Motor<T> B(B_);
      Motor<T> X(X_arr);
      Motor<T> Z(Z_arr);

      Point<T> p(p_);

      Point<T> p1 = p.spin(A * X);
      Point<T> p2 = p.spin(Z * B);

      for (int i = 0; i < 3; ++i) residuals[i] = p1[i] - p2[i];

      return true;
    }

   private:
    Mot A_;
    Mot B_;
    Pnt p_;
  };

  struct HandEyeReprojectionError {
    HandEyeReprojectionError(const Mot &A, const Pnt &point,
                             const Vector<double> &pixel)
        : A_(A), point_(point), pixel_(pixel) {}

    // Apply camera intrinsics to the normalized point to get image coordinates.
    // This applies the radial lens distortion to a point which is in normalized
    // camera coordinates (i.e. the principal point is at (0, 0)) to get image
    // coordinates in pixels. Templated for use with autodifferentiation.
    template <typename T>
    inline static void ApplyRadialDistortionCameraIntrinsics(
        const T &focal_length_x, const T &focal_length_y,
        const T &principal_point_x, const T &principal_point_y, const T &k1,
        const T &k2, const T &k3, const T &p1, const T &p2,
        const T &normalized_x, const T &normalized_y, T *image_x, T *image_y) {
      T x = normalized_x;
      T y = normalized_y;

      // Apply distortion to the normalized points to get (xd, yd).
      T r2 = x * x + y * y;
      T r4 = r2 * r2;
      T r6 = r4 * r2;
      T r_coeff = (T(1) + k1 * r2 + k2 * r4 + k3 * r6);
      T xd = x * r_coeff + T(2) * p1 * x * y + p2 * (r2 + T(2) * x * x);
      T yd = y * r_coeff + T(2) * p2 * x * y + p1 * (r2 + T(2) * y * y);

      // Apply focal length and principal point to get the final image
      // coordinates.
      *image_x = focal_length_x * xd + principal_point_x;
      *image_y = focal_length_y * yd + principal_point_y;
    }

    template <typename T>
    auto operator()(const T *const X_arr, const T *const Z_arr,
                    const T *const intrinsics_arr, T *residuals) const -> bool {
      // Unpack the intrinsics.
      const T &focal_length_x = intrinsics_arr[OFFSET_FOCAL_LENGTH_X];
      const T &focal_length_y = intrinsics_arr[OFFSET_FOCAL_LENGTH_Y];
      const T &principal_point_x = intrinsics_arr[OFFSET_PRINCIPAL_POINT_X];
      const T &principal_point_y = intrinsics_arr[OFFSET_PRINCIPAL_POINT_Y];
      const T &k1 = intrinsics_arr[OFFSET_K1];
      const T &k2 = intrinsics_arr[OFFSET_K2];
      const T &k3 = intrinsics_arr[OFFSET_K3];
      const T &p1 = intrinsics_arr[OFFSET_P1];
      const T &p2 = intrinsics_arr[OFFSET_P2];

      Motor<T> X(X_arr);
      Motor<T> Z(Z_arr);
      Motor<T> A(A_);
      Point<T> point(point_);
      // Point<T> point2 = point.spin(Z.reverse() * A * X);
      Point<T> point2 = point.spin(X * A * Z.reverse());

      // Compute normalized coordinates: x /= x[2].
      T xn = point2[0] / point2[2];
      T yn = point2[1] / point2[2];

      T predicted_x, predicted_y;

      // Apply distortion to the normalized points to get (xd, yd).
      // TODO(keir): Do early bailouts for zero distortion; these are expensive
      // jet operations.
      ApplyRadialDistortionCameraIntrinsics(
          focal_length_x, focal_length_y, principal_point_x, principal_point_y,
          k1, k2, k3, p1, p2, xn, yn, &predicted_x, &predicted_y);

      // The error is the difference between the predicted and observed
      // position.
      residuals[0] = predicted_x - T(pixel_[0]);
      residuals[1] = predicted_y - T(pixel_[1]);

      return true;
    }

   private:
    Mot A_;
    Pnt point_;
    Vector<double> pixel_;
  };

  auto AddResidualBlock(const Mot &A, const Pnt &point,
                        const Vector<double> &pixel) -> bool {
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<HandEyeReprojectionError, 2, 8, 8, 9>(
            new HandEyeReprojectionError(A, point, pixel));
    problem_.AddResidualBlock(cost_function, NULL, &X_[0], &Z_[0],
                              &intrinsics_[0]);
    return true;
  }

  auto AddResidualBlock2(const Mot &A, const Mot &B, const Pnt &point) -> bool {
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<AXZBError, 3, 8, 8>(
            new AXZBError(A, B, point));
    problem_.AddResidualBlock(cost_function, NULL, &X_[0], &Z_[0]);
    return true;
  }

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
    problem_.SetParameterBlockConstant(&intrinsics_[0]);
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
      .def("add_residual_block", &HandEyeCalibrator::AddResidualBlock)
      .def("add_residual_block_2", &HandEyeCalibrator::AddResidualBlock2)
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
