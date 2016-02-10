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

class MotorEstimationSolver {
 public:
  MotorEstimationSolver() {}
  MotorEstimationSolver(const MotorEstimationSolver &motor_estimation_solver) {}
  MotorEstimationSolver(const Mot &motor, const py::dict &solver_options)
      : motor_(motor) {
    SetSolverOptions(solver_options, options_);
  }

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
      .def(py::init<const Mot &, const py::dict &>())
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
          "trust_region_minimizer_iterations_to_dump",
          [](MotorEstimationSolver &instance) {
            return instance.options_.trust_region_minimizer_iterations_to_dump;
          },
          [](MotorEstimationSolver &instance, std::vector<int> arg) {
            std::vector<int> trust_region_minimizer_iterations_to_dump{};
            for (auto i : arg) {
              trust_region_minimizer_iterations_to_dump.push_back(i);
            }
            instance.options_.trust_region_minimizer_iterations_to_dump.resize(
                trust_region_minimizer_iterations_to_dump.size());
            for (int i = 0; i < trust_region_minimizer_iterations_to_dump.size(); ++i)
              instance.options_.trust_region_minimizer_iterations_to_dump[i] = 2;
          })

      ;

  return m.ptr();
}

}  // namespace game
