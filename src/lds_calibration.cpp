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
using vsr::cga::FlatPoint;
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
using vsr::cga::Tnv;

using vsr::nga::Op;

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

  LaserDistanceSensorCalibrator(const Vec& point, const Vec& direction,
                                const bp::dict& solver_options)
      : point_(point), direction_(direction) {
    SetSolverOptions(solver_options, options_);
  }

  LaserDistanceSensorCalibrator(const Vec& point, const double theta,
                                const double phi,
                                const bp::dict& solver_options)
      : point_(point), theta_(theta), phi_(phi) {
    SetSolverOptions(solver_options, options_);
  }

  struct AnglesDistanceErrorCostFunctor {
    AnglesDistanceErrorCostFunctor(const Mot& forward_kinematics,
                                   const Dlp& measurement_plane,
                                   const double measurement_distance)
        : forward_kinematics_(forward_kinematics),
          measurement_plane_(measurement_plane),
          measurement_distance_(measurement_distance) {}

    template <typename T>
    auto operator()(const T* const point, const T* const theta,
                    const T* const phi, T* residual) const -> bool {
      Motor<T> M_forward_kinematics(forward_kinematics_);
      DualPlane<T> dlp(measurement_plane_);

      Vector<T> l0 = Vector<T>(point).null().spin(M_forward_kinematics);

      Vector<T> direction = Vector<T>(T(1.0), T(0.0), T(0.0));
      direction = direction.spin(Rotor<T>(
          cos(theta[0] / T(2.0)), sin(theta[0] / T(2.0)), T(0.0), T(0.0)));
      // direction = direction.spin(
      // Rotor<T>(cos(phi[0] / T(2.0)), T(0.0), T(0.0), sin(phi[0] / T(2.0))));

      auto l = (Vector<T>(direction).unit() ^ Infinity<T>(T(1.0)))
                   .spin(M_forward_kinematics);

      Vector<T> n(dlp[0], dlp[1], dlp[2]);
      Vector<T> p0(dlp[0] * dlp[3], dlp[1] * dlp[3], dlp[2] * dlp[3]);
      Vector<T> p0l0 = p0 - l0;
      T num = ((p0 - l0) <= n)[0];
      T den = (Vector<T>(l[0], l[1], l[2]) <= n)[0];
      T distance = num / den;
      residual[0] = T(measurement_distance_) - distance;

      return true;
    }

   private:
    const Mot forward_kinematics_;
    const Dlp measurement_plane_;
    const double measurement_distance_;
  };

  struct PointDirectionDistanceErrorCostFunctor {
    PointDirectionDistanceErrorCostFunctor(const Mot& forward_kinematics,
                                           const Dlp& measurement_plane,
                                           const double measurement_distance)
        : forward_kinematics_(forward_kinematics),
          measurement_plane_(measurement_plane),
          measurement_distance_(measurement_distance) {}

    template <typename T>
    auto operator()(const T* const point, const T* const direction,
                    T* residual) const -> bool {
      Motor<T> M_forward_kinematics(forward_kinematics_);
      DualPlane<T> dlp(measurement_plane_);

      Vector<T> l0 = Vector<T>(point).null().spin(M_forward_kinematics);

      auto l = (Vector<T>(direction).unit() ^ Infinity<T>(T(1.0)))
                   .spin(M_forward_kinematics);

      // DualLine<T> dll = (p0.null() ^ l0.unit() ^ Infinity<T>(T(1.0)))
      //.dual()
      //.spin(M_forward_kinematics);

      // FlatPoint<T> flp = (dll ^ dlp).dual();

      // DualSphere<T> s = (p0.null() <= flp);
      // T distance = sqrt((s <= s)[0]);

      // std::cout << distance << std::endl;

      Vector<T> n(dlp[0], dlp[1], dlp[2]);
      Vector<T> p0(dlp[0] * dlp[3], dlp[1] * dlp[3], dlp[2] * dlp[3]);
      Vector<T> p0l0 = p0 - l0;
      T num = ((p0 - l0) <= n)[0];
      T den = (Vector<T>(l[0], l[1], l[2]) <= n)[0];
      T distance = num / den;
      residual[0] = T(measurement_distance_) - distance;

      return true;
    }

   private:
    const Mot forward_kinematics_;
    const Dlp measurement_plane_;
    const double measurement_distance_;
  };

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

  struct DirectionNormalizationParameterization {
    template <typename T>
    bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
      std::cout << "Direction Normalization" << std::endl;
      x_plus_delta[0] = x[0] + delta[0];
      x_plus_delta[1] = x[1] + delta[1];
      x_plus_delta[2] = x[2] + delta[2];
      T scale = static_cast<T>(1.0) / sqrt(x_plus_delta[0] * x_plus_delta[0] +
                                           x_plus_delta[1] * x_plus_delta[1] +
                                           x_plus_delta[2] * x_plus_delta[2]);
      x_plus_delta[0] *= scale;
      x_plus_delta[1] *= scale;
      x_plus_delta[2] *= scale;
      return true;
    }
  };

  auto Summary() const -> bp::dict { return game::SummaryToDict(summary_); }

  auto AddTangentVectorResidualBlock(const Mot& forward_kinematics,
                                     const Dlp& measurement_plane,
                                     const double measurement_distance)
      -> bool {
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<MeasurementDistanceErrorCostFunctor, 1,
                                        8>(
            new MeasurementDistanceErrorCostFunctor(
                forward_kinematics, measurement_plane, measurement_distance));
    problem_.AddResidualBlock(cost_function, NULL, &motor_[0]);
    return true;
  }

  auto AddPointDirectionResidualBlock(const Mot& forward_kinematics,
                                      const Dlp& measurement_plane,
                                      const double measurement_distance)
      -> bool {
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<PointDirectionDistanceErrorCostFunctor,
                                        1, 3, 3>(
            new PointDirectionDistanceErrorCostFunctor(
                forward_kinematics, measurement_plane, measurement_distance));
    problem_.AddResidualBlock(cost_function, NULL, &point_[0], &direction_[0]);
    return true;
  }

  auto AddAngleDistanceResidualBlock(const Mot& forward_kinematics,
                                     const Dlp& measurement_plane,
                                     const double measurement_distance)
      -> bool {
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<AnglesDistanceErrorCostFunctor, 1, 3, 1,
                                        1>(new AnglesDistanceErrorCostFunctor(
            forward_kinematics, measurement_plane, measurement_distance));
    problem_.AddResidualBlock(cost_function, NULL, &point_[0], &theta_, &phi_);
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

  auto SetDirectionParameterizationTypeFromString(const std::string& type)
      -> bool {
    if (type == "NORMALIZE") {
      problem_.SetParameterization(
          &direction_[0], new ceres::AutoDiffLocalParameterization<
                              DirectionNormalizationParameterization, 3, 3>);
      return true;
    }
  }

  auto Solve() -> bp::tuple {
    ceres::Solve(options_, &problem_, &summary_);
    return bp::make_tuple(motor_, Summary());
  }

  auto SolvePointDirection() -> bp::tuple {
    ceres::Solve(options_, &problem_, &summary_);
    return bp::make_tuple(point_, direction_, Summary());
  }

  auto SolvePointAngles() -> bp::tuple {
    ceres::Solve(options_, &problem_, &summary_);
    return bp::make_tuple(point_, theta_, phi_, Summary());
  }

 private:
  Mot motor_;
  Vec point_;
  Vec direction_;
  double theta_;
  double phi_;

  ceres::Problem problem_;
  ceres::Solver::Options options_;
  ceres::Solver::Summary summary_;
};

BOOST_PYTHON_MODULE_INIT(liblds_calibration) {
  bp::class_<LaserDistanceSensorCalibrator>(
      "LaserDistanceSensorCalibrator", bp::init<const Mot&, const bp::dict&>())
      .def(bp::init<const Vec&, const Vec&, const bp::dict&>())
      .def(bp::init<const Vec&, const double, const double, const bp::dict&>())
      .def("add_tangent_vector_residual_block",
           &LaserDistanceSensorCalibrator::AddTangentVectorResidualBlock)
      .def("add_point_direction_residual_block",
           &LaserDistanceSensorCalibrator::AddPointDirectionResidualBlock)
      .def("add_point_angles_residual_block",
           &LaserDistanceSensorCalibrator::AddAngleDistanceResidualBlock)
      .def("set_parameterization", &LaserDistanceSensorCalibrator::
                                       SetMotorParameterizationTypeFromString)
      .def("set_direction_parameterization",
           &LaserDistanceSensorCalibrator::
               SetDirectionParameterizationTypeFromString)
      .def("solve", &LaserDistanceSensorCalibrator::Solve)
      .def("solve_point_angles",
           &LaserDistanceSensorCalibrator::SolvePointAngles)
      .def("solve_point_direction",
           &LaserDistanceSensorCalibrator::SolvePointDirection);

  bp::def("transform_tangent_vector", &TransformTangentVector);
}

}  // namespace game
