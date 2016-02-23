//
// Created by lars on 17.11.15.
//

#include <pybind11/pybind11.h>

#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>

#include "game/vsr/vsr.h"
#include "game/adept_autodiff_cost_function.h"
#include "game/adept_autodiff_local_parameterization.h"
#include "game/motor_parameterization.h"
#include "game/ceres_python_utils.h"

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
    options_.trust_region_minimizer_iterations_to_dump =
        std::vector<int>{0, 1, 2, 3, 4};
    options_.trust_region_problem_dump_directory =
        std::string{"/Users/lars/devel/game/dump"};
  }

  struct TangentVectorPointAngleErrorCostFunctor {
    TangentVectorPointAngleErrorCostFunctor(const Tnv &a, const Tnv &b)
        : a_(a), b_(b) {}

    template <typename T>
    bool operator()(const T *const motor, T *residual) const {
      Motor<T> M(motor);
      TangentVector<T> a(a_);
      TangentVector<T> b(b_);
      TangentVector<T> c = a.spin(M);

      Point<T> pb = b / (-Infinity<T>(T(1.0)) <= b);
      Point<T> pc = c / (-Infinity<T>(T(1.0)) <= c);

      for (int i = 0; i < 3; ++i) {
        residual[i] = pc[i] - pb[i];
      }

      DirectionVector<T> db = -(Infinity<T>(T(1.0)) <= b) ^ Infinity<T>(T(1.0));
      DirectionVector<T> dc = -(Infinity<T>(T(1.0)) <= c) ^ Infinity<T>(T(1.0));

      // residual[3] = (Vector<T>{db[0], db[1], db[2]} <= Vector<T>{dc[0],
      // dc[1], dc[2]})[0];

      // for (int i = 3; i < 6; ++i) {
      //   residual[i] = dc[i] - db[i];
      // }

      return true;
    }

   private:
    const Tnv a_;
    const Tnv b_;
  };

  struct DualPlaneAngleErrorCostFunctor {
    DualPlaneAngleErrorCostFunctor(const Dlp &a, const Dlp &b) : a_(a), b_(b) {}

    template <typename T>
    bool operator()(const T *const motor, T *residual) const {
      Motor<T> M(motor);
      DualPlane<T> a(a_);
      DualPlane<T> b(b_);
      DualPlane<T> c = a.spin(M);

      Origin<T> no{T(1.0)};
      Infinity<T> ni{T(1.0)};

      Motor<T> X = Scalar<T>{0.5} * (c / b);
      Rotor<T> R{X[0], X[1], X[2], X[3]};
      Vector<T> t = Scalar<T>{-2.0} * (no <= X) / R;
      T distance = t.norm();

      // T distance;
      // if (abs(T(1.0) - X[0]) > T(0.0)) {
      //   Bivector<T> B{R[1], R[2], R[3]};
      //   B = B.unit();
      //   Vector<T> w = Op::reject(t, B);
      //   distance = w.norm();
      //   residual[0] = distance;
      // } else {
      //   distance = t.norm();
      //   residual[0] = distance;
      // }

      // residual[1] = T(1.0) - X[0];

      Scalar<T> cos_theta = c <= b;

      residual[0] = X[0];
      residual[1] = distance;

      return true;
    }

   private:
    const Dlp a_;
    const Dlp b_;
  };

  struct LineAngleDistanceNormCostFunctor {
    LineAngleDistanceNormCostFunctor(const Dll &a, const Dll &b)
        : a_(a), b_(b) {}

    template <typename T>
    auto operator()(const T *const motor, T *residual) const -> bool {
      Motor<T> M(motor);
      DualLine<T> a(a_);
      DualLine<T> b(b_);
      DualLine<T> c = a.spin(M);

      Origin<T> no{T(1.0)};
      Infinity<T> ni{T(1.0)};

      Motor<T> X = Scalar<T>{0.5} * (c / b);
      Rotor<T> R{X[0], X[1], X[2], X[3]};
      Vector<T> t = Scalar<T>{-2.0} * (no <= X) / R;

      T distance;
      if (abs(T(1.0) - X[0]) > T(0.0)) {
        Bivector<T> B{R[1], R[2], R[3]};
        B = B.unit();
        Vector<T> w = Op::reject(t, B);
        distance = w.norm();
        residual[0] = distance;
      } else {
        distance = t.norm();
        residual[0] = distance;
      }

      residual[1] = T(1.0) - X[0];

      return true;
    }

   private:
    const Dll a_;
    const Dll b_;
  };

  struct LineAngleDistanceCostFunctor {
    LineAngleDistanceCostFunctor(const Dll &a, const Dll &b) : a_(a), b_(b) {}

    template <typename T>
    auto operator()(const T *const motor, T *residual) const -> bool {
      Motor<T> M(motor);
      DualLine<T> a(a_);
      DualLine<T> b(b_);
      DualLine<T> c = a.spin(M);

      Origin<T> no{T(1.0)};
      Infinity<T> ni{T(1.0)};

      Motor<T> X = Scalar<T>{0.5} * (c / b);
      Rotor<T> R{X[0], X[1], X[2], X[3]};
      Vector<T> t = Scalar<T>{-2.0} * (no <= X) / R;

      T scale{T(1.0)};
      if (abs(T(1.0) - X[0]) > T(0.0)) {
        Bivector<T> B{R[1], R[2], R[3]};
        B = B.unit();
        Vector<T> w = Op::reject(t, B);
        residual[0] = w[0];
        residual[1] = w[1];
        residual[2] = w[2];
      } else {
        residual[0] = t[0];
        residual[1] = t[1];
        residual[2] = t[2];
      }

      residual[3] = T(1.0) - X[0];

      return true;
    }

   private:
    const Dll a_;
    const Dll b_;
  };

  struct LineCorrespondencesCostFunctor {
    LineCorrespondencesCostFunctor(const Dll &a, const Dll &b) : a_(a), b_(b) {}

    template <typename T>
    auto operator()(const T *const motor, T *residual) const -> bool {
      Motor<T> M(motor);
      DualLine<T> a(a_);
      DualLine<T> b(b_);
      DualLine<T> c = a.spin(M);

      for (int i = 0; i < b.Num; ++i) {
        residual[i] = c[i] - b[i];
      }

      return true;
    }

   private:
    const Dll a_;
    const Dll b_;
  };

  struct PointDistanceCostFunctor {
    PointDistanceCostFunctor(const Pnt &a, const Pnt &b) : a_(a), b_(b) {}

    template <typename T>
    auto operator()(const T *const motor, T *residual) const -> bool {
      Motor<T> M(motor);
      Point<T> a(a_);
      Point<T> b(b_);
      Point<T> c = a.spin(M);

      // T cos_distance = cos(sqrt(T(-2.0) * minus_half_distance_squared));
      // residual[0] = cos_distance;

      T minus_half_distance_squared = (c <= b)[0];
      residual[0] = minus_half_distance_squared;

      return true;
    }

   private:
    const Pnt a_;
    const Pnt b_;
  };

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

  auto Summary() const -> py::dict { return game::SummaryToDict(summary_); }

  bool AddTangentVectorPointAngleErrorResidualBlock(const Tnv &a,
                                                    const Tnv &b) {
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<TangentVectorPointAngleErrorCostFunctor,
                                        3, 8>(
            new TangentVectorPointAngleErrorCostFunctor(a, b));
    problem_.AddResidualBlock(cost_function, NULL, &motor_[0]);
    return true;
  }

  bool AddDualPlaneAngleErrorResidualBlock(const Dlp &a, const Dlp &b) {
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<DualPlaneAngleErrorCostFunctor, 2, 8>(
            new DualPlaneAngleErrorCostFunctor(a, b));
    problem_.AddResidualBlock(cost_function, NULL, &motor_[0]);
    return true;
  }

  auto AddLineCorrespondencesResidualBlock(const Dll &a, const Dll &b) -> bool {
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<LineCorrespondencesCostFunctor, 6, 8>(
            new LineCorrespondencesCostFunctor(a, b));
    problem_.AddResidualBlock(cost_function, NULL, &motor_[0]);
    return true;
  }

  auto AddLineAngleDistanceResidualBlock(const Dll &a, const Dll &b) -> bool {
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<LineAngleDistanceCostFunctor, 4, 8>(
            new LineAngleDistanceCostFunctor(a, b));
    problem_.AddResidualBlock(cost_function, NULL, &motor_[0]);
    return true;
  }

  auto AddLineAngleDistanceNormResidualBlock(const Dll &a, const Dll &b)
      -> bool {
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<LineAngleDistanceNormCostFunctor, 2, 8>(
            new LineAngleDistanceNormCostFunctor(a, b));
    problem_.AddResidualBlock(cost_function, NULL, &motor_[0]);
    return true;
  }

  auto AddPointCorrespondencesResidualBlock(const Pnt &a, const Pnt &b)
      -> bool {
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<PointCorrespondencesCostFunctor, 3, 8>(
            new PointCorrespondencesCostFunctor(a, b));
    problem_.AddResidualBlock(cost_function, NULL, &motor_[0]);
    return true;
  }

  auto AddAdeptPointCorrespondencesResidualBlock(const Pnt &a, const Pnt &b)
      -> bool {
    ceres::CostFunction *cost_function =
        new AdeptAutoDiffCostFunction<PointCorrespondencesCostFunctor, 3, 8>(
            new PointCorrespondencesCostFunctor(a, b));
    problem_.AddResidualBlock(cost_function, NULL, &motor_[0]);
    return true;
  }

  auto AddPointDistanceResidualBlock(const Pnt &a, const Pnt &b) -> bool {
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<PointDistanceCostFunctor, 1, 8>(
            new PointDistanceCostFunctor(a, b));
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
    } else if (type == "BIVECTOR_GENERATOR_ADEPT") {
      std::cout << "game:: ADEPT Using bivector generator (Versor)."
                << std::endl;
      problem_.SetParameterization(
          &motor_[0],
          new AdeptAutoDiffLocalParameterization<MotorFromBivectorGenerator, 8,
                                                 6>);
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

PYBIND11_PLUGIN(motor_estimation) {
  py::module m("motor_estimation", "motor estimation");

  py::class_<MotorEstimationSolver>(m, "MotorEstimationSolver")
      .def(py::init<const Mot &>())
      .def("add_tangent_vector_point_angle_error_residual_block",
           &MotorEstimationSolver::AddTangentVectorPointAngleErrorResidualBlock)
      .def("add_dual_plane_angle_error_residual_block",
           &MotorEstimationSolver::AddDualPlaneAngleErrorResidualBlock)
      .def("add_line_correspondences_residual_block",
           &MotorEstimationSolver::AddLineCorrespondencesResidualBlock)
      .def("add_line_angle_distance_residual_block",
           &MotorEstimationSolver::AddLineAngleDistanceResidualBlock)
      .def("add_line_angle_distance_norm_residual_block",
           &MotorEstimationSolver::AddLineAngleDistanceNormResidualBlock)
      .def("add_point_correspondences_residual_block",
           &MotorEstimationSolver::AddPointCorrespondencesResidualBlock)
      .def("add_adept_point_correspondences_residual_block",
           &MotorEstimationSolver::AddAdeptPointCorrespondencesResidualBlock)
      .def("add_point_distance_residual_block",
           &MotorEstimationSolver::AddPointDistanceResidualBlock)
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

}  // namespace game
