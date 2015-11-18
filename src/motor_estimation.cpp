//
// Created by lars on 17.11.15.
//

#include <iostream>
#include <boost/numpy.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>

#include "game/vsr/vsr.h"
#include "game/motor_parameterization.h"
#include "game/ceres_python_utils.h"

namespace bp = boost::python;
namespace np = boost::numpy;

using vsr::cga::Point;
using vsr::cga::DualLine;
using vsr::cga::Motor;
using vsr::cga::Pnt;
using vsr::cga::Dll;
using vsr::cga::Mot;

namespace game {

class MotorEstimationSolver {
public:
  MotorEstimationSolver() {}
  MotorEstimationSolver(const MotorEstimationSolver& motor_estimation_solver) {}
  MotorEstimationSolver(const Mot& motor, const bp::dict& solver_options)
      : motor_(motor) {
    SetSolverOptions(solver_options, options_);
  }

  struct LineCorrespondencesCostFunctor {
    LineCorrespondencesCostFunctor(const Dll& a, const Dll& b) : a_(a), b_(b) {}

    template <typename T>
    auto operator()(const T* const motor, T* residual) const -> bool {
      Motor<T> M(motor);
      DualLine<T> a(a_);
      DualLine<T> b(b_);
      DualLine<T> c = a.spin(M);

      for (int i = 0; i < b.Num; ++i) {
        residual[i] = c[i] - b[i];
      }

      // Using just angle between lines does not seem to work ...
      // T theta = acos((c <= b)[0]);
      // residual[0] = theta;

      return true;
    }

  private:
    const Dll a_;
    const Dll b_;
  };

  struct PointCorrespondencesCostFunctor {
    PointCorrespondencesCostFunctor(const Pnt& a, const Pnt& b)
        : a_(a), b_(b) {}

    template <typename T>
    auto operator()(const T* const motor, T* residual) const -> bool {
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

  auto Summary() const -> bp::dict { return game::SummaryToDict(summary_); }

  auto AddLineCorrespondencesResidualBlock(const Dll& a, const Dll& b) -> bool {
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<LineCorrespondencesCostFunctor, 6, 8>(
            new LineCorrespondencesCostFunctor(a, b));
    problem_.AddResidualBlock(cost_function, NULL, &motor_[0]);
    return true;
  }

  auto AddPointCorrespondencesResidualBlock(const Pnt& a, const Pnt& b)
      -> bool {
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<PointCorrespondencesCostFunctor, 3, 8>(
            new PointCorrespondencesCostFunctor(a, b));
    problem_.AddResidualBlock(cost_function, NULL, &motor_[0]);
    return true;
  }

  auto SetMotorParameterizationTypeFromString(const std::string& type) -> void {
    if (type == "NORMALIZE") {
      problem_.SetParameterization(
          &motor_[0],
          new ceres::AutoDiffLocalParameterization<MotorNormalizeRotorPlus, 8,
                                                   8>);
    } else if (type == "POLAR_DECOMPOSITION") {
      std::cout << "game:: Using polar decomposition" << std::endl;
      problem_.SetParameterization(
          &motor_[0],
          new ceres::AutoDiffLocalParameterization<MotorPolarDecomposition, 8, 8>);
    } else if (type == "BIVECTOR_GENERATOR") {
      std::cout << "game:: Using bivector generator, hand written" << std::endl;
      problem_.SetParameterization(
          &motor_[0],
          new ceres::AutoDiffLocalParameterization<MotorFromBivectorGenerator, 8, 6>);
    } else {
      std::cout << "Unknown motor parameterization type" << std::endl;
    }
  }

  auto Solve() -> bp::tuple {
    ceres::Solve(options_, &problem_, &summary_);
    return bp::make_tuple(motor_, Summary());
  }

private:
  Mot motor_;

  ceres::Problem problem_;
  ceres::Solver::Options options_;
  ceres::Solver::Summary summary_;
};

BOOST_PYTHON_MODULE_INIT(libmotor_estimation) {
  np::initialize();

  bp::class_<MotorEstimationSolver>("MotorEstimationSolver",
                                    bp::init<const Mot&, const bp::dict&>())
      .def("add_line_correspondences_residual_block",
           &MotorEstimationSolver::AddLineCorrespondencesResidualBlock)
      .def("add_point_correspondences_residual_block",
           &MotorEstimationSolver::AddPointCorrespondencesResidualBlock)
      .def("set_parameterization",
           &MotorEstimationSolver::SetMotorParameterizationTypeFromString)
      .def("solve", &MotorEstimationSolver::Solve);
}

}  // namespace game
