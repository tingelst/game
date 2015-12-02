// Geometric Algebra Multivector Estimation Library
// Copyright (c) 2015 Lars Tingelstad <lars.tingeltad@ntnu.no>
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// Author: Lars Tingelstad

#include <boost/python.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>

#include "game/vsr/vsr.h"
#include "game/adept_autodiff_cost_function.h"
#include "game/adept_autodiff_local_parameterization.h"
#include "game/motor_parameterization.h"
#include "game/ceres_python_utils.h"

namespace bp = boost::python;

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
using vsr::cga::TangentVector;
using vsr::cga::DualPlane;
using vsr::cga::PointPair;
using vsr::cga::DualSphere;

using vsr::cga::Vec;
using vsr::cga::Par;
using vsr::cga::Pnt;
using vsr::cga::Dlp;
using vsr::cga::Dll;
using vsr::cga::Mot;
using vsr::cga::Ori;
using vsr::cga::Inf;
using vsr::cga::Con;
using vsr::nga::Op;
using vsr::cga::Tnv;

namespace game {

auto TransformTangentVector(const Mot& mot, const Tnv& tnv) -> Par {
  return mot * tnv * ~mot;
}

class LaserDistanceSensorCalibrator {
 public:
  LaserDistanceSensorCalibrator() {}
  LaserDistanceSensorCalibrator(
      const LaserDistanceSensorCalibrator& lds_calibrator) {}
  LaserDistanceSensorCalibrator(const Mot& motor,
                                const bp::dict& solver_options)
      : motor_(motor) {
    SetSolverOptions(solver_options, options_);
  }

  struct MeasurementDistanceErrorCostFunctor {
    MeasurementDistanceErrorCostFunctor(const Mot& forward_kinematics,
                                        const Dlp& measurement_plane,
                                        const double measurement_distance)
        : forward_kinematics_(forward_kinematics),
          measurement_plane_(measurement_plane),
          measurement_distance_(measurement_distance) {}

    template <typename T>
    auto operator()(const T* const motor, T* residual) const -> bool {
      Motor<T> M_laser_in_ee(motor);
      Motor<T> M_forward_kinematics(forward_kinematics_);
      DualPlane<T> dlp(measurement_plane_);

      Motor<T> M = M_forward_kinematics * M_laser_in_ee;

      PointPair<T> t =
          M * (Origin<T>(T(1.0)) ^ Vector<T>(T(1.0), T(0.0), T(0.0))) * ~M;

      DualSphere<T> s = (dlp <= t) / (Infinity<T>(T(1.0)) <= (dlp <= t));

      T radius = sqrt((s <= s)[0]);

      residual[0] = T(measurement_distance_) - radius;

      return true;
    }

   private:
    const Mot forward_kinematics_;
    const Dlp measurement_plane_;
    const double measurement_distance_;
  };

  auto Summary() const -> bp::dict { return game::SummaryToDict(summary_); }

  auto AddResidualBlock(const Mot& forward_kinematics,
                        const Dlp& measurement_plane,
                        const double measurement_distance) -> bool {
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<MeasurementDistanceErrorCostFunctor, 1,
                                        8>(
            new MeasurementDistanceErrorCostFunctor(
                forward_kinematics, measurement_plane, measurement_distance));
    problem_.AddResidualBlock(cost_function, NULL, &motor_[0]);
    return true;
  }

  auto SetMotorParameterizationTypeFromString(const std::string& type) -> void {
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

BOOST_PYTHON_MODULE_INIT(liblds_calibration) {
  bp::class_<LaserDistanceSensorCalibrator>(
      "LaserDistanceSensorCalibrator", bp::init<const Mot&, const bp::dict&>())
      .def("add_residual_block",
           &LaserDistanceSensorCalibrator::AddResidualBlock)
      .def("set_parameterization", &LaserDistanceSensorCalibrator::
                                       SetMotorParameterizationTypeFromString)
      .def("solve", &LaserDistanceSensorCalibrator::Solve);

  bp::def("transform_tangent_vector", &TransformTangentVector);
}

}  // namespace game
