// sphere_fit.cpp
//
// Example from Foundations of Geometric Algebra Computing
// by Dietmar Hildenbrand
//
// Page 68.

#include <iostream>
#include <string>
#include <boost/numpy.hpp>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <hep/ga.hpp>

#include "game/types.h"
#include "game/ceres_python_utils.h"

namespace bp = boost::python;
namespace np = boost::numpy;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;

namespace game {

struct SphereFit {
  SphereFit(const SphereFit &sphere_fit) {}
  SphereFit() {}

  void SetSolverOptions(const bp::dict &solver_options) {
    bp::extract<std::string> linear_solver_type(
        solver_options["linear_solver_type"]);
    if (linear_solver_type.check()) {
      ceres::StringToLinearSolverType(linear_solver_type(),
                                      &options_.linear_solver_type);
    }

    bp::extract<int> max_num_iterations(solver_options["max_num_iterations"]);
    if (max_num_iterations.check()) {
      options_.max_num_iterations = max_num_iterations();
    }

    bp::extract<int> num_threads(solver_options["num_threads"]);
    if (num_threads.check()) {
      options_.num_threads = num_threads();
    }

    bp::extract<int> num_linear_solver_threads(
        solver_options["num_linear_solver_threads"]);
    if (num_linear_solver_threads.check()) {
      options_.num_linear_solver_threads = num_linear_solver_threads();
    }

    bp::extract<double> parameter_tolerance(
        solver_options["parameter_tolerance"]);
    if (parameter_tolerance.check()) {
      options_.parameter_tolerance = parameter_tolerance();
    }

    bp::extract<double> function_tolerance(
        solver_options["function_tolerance"]);
    if (function_tolerance.check()) {
      options_.function_tolerance = function_tolerance();
    }

    bp::extract<std::string> trust_region_strategy_type(
        solver_options["trust_region_strategy_type"]);
    if (trust_region_strategy_type.check()) {
      ceres::StringToTrustRegionStrategyType(
          trust_region_strategy_type(), &options_.trust_region_strategy_type);
    }

    bp::extract<bool> minimizer_progress_to_stdout(
        solver_options["minimizer_progress_to_stdout"]);
    if (minimizer_progress_to_stdout.check()) {
      options_.minimizer_progress_to_stdout = minimizer_progress_to_stdout();
    }

    bp::extract<std::string> minimizer_type(solver_options["minimizer_type"]);
    if (minimizer_type.check()) {
      ceres::StringToMinimizerType(minimizer_type(), &options_.minimizer_type);
    }

    bp::extract<bp::list> trust_region_minimizer_iterations_to_dump(
        solver_options["trust_region_minimizer_iterations_to_dump"]);
    if (trust_region_minimizer_iterations_to_dump.check()) {
      std::vector<int> iterations_to_dump{};
      bp::list list = trust_region_minimizer_iterations_to_dump();
      for (int i = 0; i < bp::len(list); ++i) {
        iterations_to_dump.push_back(bp::extract<int>(list[i])());
      }
      options_.trust_region_minimizer_iterations_to_dump = iterations_to_dump;
    }

    bp::extract<std::string> trust_region_problem_dump_directory(
        solver_options["trust_region_problem_dump_directory"]);
    if (trust_region_problem_dump_directory.check()) {
      options_.trust_region_problem_dump_directory =
          trust_region_problem_dump_directory();
    }
  }

  template <typename T, typename MultivectorT>
  static auto Inverse(const MultivectorT &multivector) -> MultivectorT {
    return cga::Scalar<T>{static_cast<T>(1.0) /
                          hep::eval((multivector * ~multivector))[0]} *
           ~multivector;
  }

  struct CostFunctor {
    CostFunctor(const double *point) : point_(point) {}

    template <typename T>
    bool operator()(const T *const s /* sphere: 5 parameters */,
                    T *residual /* 1 parameter */) const {
      // Conformal split
      cga::Infty<T> ni{static_cast<T>(-1.0), static_cast<T>(1.0)};
      cga::Orig<T> no{static_cast<T>(0.5), static_cast<T>(0.5)};

      // Euclidean basis
      cga::E1<T> e1{static_cast<T>(1.0)};
      cga::E2<T> e2{static_cast<T>(1.0)};
      cga::E3<T> e3{static_cast<T>(1.0)};

      cga::Scalar<T> half{static_cast<T>(0.5)};

      // Create Euclidean point (vector)
      cga::EuclideanPoint<T> euc_point{static_cast<T>(point_[0]),
                                       static_cast<T>(point_[1]),
                                       static_cast<T>(point_[2])};

      // Create conformal point
      cga::Point<T> point =
          hep::grade<1>(euc_point + half * euc_point * euc_point * ni + no);

      // Create conformal sphere
      //      cga::Sphere<T> sphere = hep::eval(
      //          static_cast<T>(s[0] / s[3]) * e1 + static_cast<T>(s[1] / s[3])
      //          * e2 +
      //          static_cast<T>(s[2] / s[3]) * e3 + static_cast<T>(s[4] / s[3])
      //          * ni +
      //          static_cast<T>(s[3] / s[3]) * no);

      cga::Sphere<T> sphere =
          hep::eval(static_cast<T>(s[0]) * e1 + static_cast<T>(s[1]) * e2 +
                    static_cast<T>(s[2]) * e3 + static_cast<T>(s[4]) * ni +
                    static_cast<T>(s[3]) * no);

      // Evaluate distance
      //      auto distance = hep::eval(hep::inner_prod(point, sphere) *
      //                                hep::inner_prod(point,
      //                                Inverse<T>(sphere)));

      auto distance = hep::eval(hep::inner_prod(point, sphere));
      auto rho_squared = hep::eval(hep::inner_prod(sphere, sphere));

      residual[0] = distance[0] / sqrt(rho_squared[0]);
      //      residual[0] = distance[0];

      return true;
    }

   private:
    const double *point_;
  };

  static ceres::CostFunction *Create(const double *point) {
    return (new ceres::AutoDiffCostFunction<SphereFit::CostFunctor, 1, 5>(
        new SphereFit::CostFunctor(point)));
  }

  np::ndarray Run(np::ndarray sphere, np::ndarray points) {
    if (!(points.get_flags() & np::ndarray::C_CONTIGUOUS)) {
      throw "input array a must be contiguous";
    }

    auto rows_points = points.shape(0);
    auto cols_points = points.shape(1);
    auto rows_sphere = sphere.shape(0);
    auto cols_sphere = sphere.shape(1);

    if (!((rows_sphere == 5) && (cols_sphere == 1))) {
      throw "parameter array must have shape (5,1)";
    }

    double *sphere_data = reinterpret_cast<double *>(sphere.get_data());
    double *points_data = reinterpret_cast<double *>(points.get_data());

    for (int i = 0; i < rows_points; ++i) {
      ceres::CostFunction *cost_function =
          SphereFit::Create(&points_data[cols_points * i]);
      problem_.AddResidualBlock(cost_function, NULL, sphere_data);
    }

    Solve(options_, &problem_, &summary_);

    return sphere;
  }

  auto Summary() -> bp::dict { return game::SummaryToDict(summary_); }

  Problem problem_;
  Solver::Options options_;
  Solver::Summary summary_;
};
}

BOOST_PYTHON_MODULE(libsphere_fit) {
  np::initialize();

  bp::class_<game::SphereFit>("SphereFit")
      .def("run", &game::SphereFit::Run)
      .def("summary", &game::SphereFit::Summary)
      .def("set_solver_options", &game::SphereFit::SetSolverOptions);
}
