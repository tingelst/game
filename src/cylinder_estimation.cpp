#include <pybind11/pybind11.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>
#include <iostream>

#include "game/adept_autodiff_cost_function.h"
#include "game/adept_autodiff_local_parameterization.h"
#include "game/ceres_python_utils.h"
#include "game/motor_parameterization.h"
#include "game/vsr/vsr.h"

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
using vsr::cga::DirectionTrivector;
using vsr::cga::Circle;
using vsr::cga::Cir;
using vsr::cga::Pnt;
using vsr::cga::Drv;
using vsr::cga::Dll;
using vsr::cga::Dlp;
using vsr::cga::Mot;
using vsr::cga::Tnv;
using vsr::nga::Op;

namespace game {

class CylinderEstimationSolver {
public:
  CylinderEstimationSolver() {}
  CylinderEstimationSolver(
      const CylinderEstimationSolver &cylinder_estimation_solver) {}
  CylinderEstimationSolver(const Dll &line, const double &radius)
      : line_(line), radius_(radius) {}

  struct CostFunctor {
    CostFunctor(const Pnt &pnt) : pnt_(pnt) {}

    template <typename T>
    bool operator()(const T *const line, const T *const radius,
                    T *residual) const {

      DualLine<T> dll(line);
      Point<T> pnt(pnt_);

      Point<T> p1 = vsr::nga::Flat::location(dll, pnt, true);
      Scalar<T> d = p1 <= pnt;

      // std::cout << sqrt(-d[0]) << std::endl;

      std::cout << radius[0] << std::endl;

      residual[0] = sqrt(-d[0]) - radius[0];

      return true;
    }

  private:
    const Pnt pnt_;
  };

  auto AddResidualBlock(const Pnt &pnt) -> bool {
    ceres::CostFunction *cost_function =
        new ceres::AutoDiffCostFunction<CostFunctor, 1, 6, 1>(
            new CostFunctor(pnt));
    problem_.AddResidualBlock(cost_function, NULL, &line_[0], &radius_);
    return true;
  }

  auto Solve() -> std::tuple<Dll, double, py::dict> {
    ceres::Solve(options_, &problem_, &summary_);
    return std::make_tuple(line_, radius_, Summary());
  }

  auto Summary() const -> py::dict { return game::SummaryToDict(summary_); }

  // private:
  double radius_;
  Dll line_;

  ceres::Problem problem_;
  ceres::Solver::Options options_;
  ceres::Solver::Summary summary_;
};

PYBIND11_PLUGIN(cylinder_estimation) {
  py::module m("cylinder_estimation", "cylinder estimation");

  py::class_<CylinderEstimationSolver>(m, "CylinderEstimationSolver")
      .def(py::init<const Dll &, const double &>())
      .def("add_residual_block", &CylinderEstimationSolver::AddResidualBlock)
      .def("solve", &CylinderEstimationSolver::Solve)
      .def_property("num_threads",
                    [](CylinderEstimationSolver &instance) {
                      return instance.options_.num_threads;
                    },
                    [](CylinderEstimationSolver &instance, int arg) {
                      instance.options_.num_threads = arg;
                    })
      .def_property("num_linear_solver_threads",
                    [](CylinderEstimationSolver &instance) {
                      return instance.options_.num_linear_solver_threads;
                    },
                    [](CylinderEstimationSolver &instance, int arg) {
                      instance.options_.num_linear_solver_threads = arg;
                    })
      .def_property("max_num_iterations",
                    [](CylinderEstimationSolver &instance) {
                      return instance.options_.max_num_iterations;
                    },
                    [](CylinderEstimationSolver &instance, int arg) {
                      instance.options_.max_num_iterations = arg;
                    })

      .def_property("parameter_tolerance",
                    [](CylinderEstimationSolver &instance) {
                      return instance.options_.parameter_tolerance;
                    },
                    [](CylinderEstimationSolver &instance, double arg) {
                      instance.options_.parameter_tolerance = arg;
                    })
      .def_property("function_tolerance",
                    [](CylinderEstimationSolver &instance) {
                      return instance.options_.function_tolerance;
                    },
                    [](CylinderEstimationSolver &instance, double arg) {
                      instance.options_.function_tolerance = arg;
                    })
      .def_property("gradient_tolerance",
                    [](CylinderEstimationSolver &instance) {
                      return instance.options_.gradient_tolerance;
                    },
                    [](CylinderEstimationSolver &instance, double arg) {
                      instance.options_.gradient_tolerance = arg;
                    })
      .def_property("minimizer_progress_to_stdout",
                    [](CylinderEstimationSolver &instance) {
                      return instance.options_.minimizer_progress_to_stdout;
                    },
                    [](CylinderEstimationSolver &instance, double arg) {
                      instance.options_.minimizer_progress_to_stdout = arg;
                    })
      .def_property(
          "minimizer_type",
          [](CylinderEstimationSolver &instance) {
            return ceres::MinimizerTypeToString(
                instance.options_.minimizer_type);
          },
          [](CylinderEstimationSolver &instance, const std::string &arg) {
            ceres::StringToMinimizerType(arg,
                                         &instance.options_.minimizer_type);
          })
      .def_property(
          "linear_solver_type",
          [](CylinderEstimationSolver &instance) {
            return ceres::LinearSolverTypeToString(
                instance.options_.linear_solver_type);
          },
          [](CylinderEstimationSolver &instance, const std::string &arg) {
            ceres::StringToLinearSolverType(
                arg, &instance.options_.linear_solver_type);
          })
      .def_property(
          "line_search_type",
          [](CylinderEstimationSolver &instance) {
            return ceres::LineSearchTypeToString(
                instance.options_.line_search_type);
          },
          [](CylinderEstimationSolver &instance, const std::string &arg) {
            ceres::StringToLineSearchType(arg,
                                          &instance.options_.line_search_type);
          },
          "Default: WOLFE\n\nChoices are ARMIJO and WOLFE (strong Wolfe "
          "conditions). Note that in order for the assumptions underlying the "
          "BFGS and LBFGS line search direction algorithms to be guaranteed to "
          "be satisifed, the WOLFE line search should be used.")
      .def_property(
          "line_search_direction_type",
          [](CylinderEstimationSolver &instance) {
            return ceres::LineSearchDirectionTypeToString(
                instance.options_.line_search_direction_type);
          },
          [](CylinderEstimationSolver &instance, const std::string &arg) {
            ceres::StringToLineSearchDirectionType(
                arg, &instance.options_.line_search_direction_type);
          })
      .def_property(
          "nonlinear_conjugate_gradient_type",
          [](CylinderEstimationSolver &instance) {
            return ceres::NonlinearConjugateGradientTypeToString(
                instance.options_.nonlinear_conjugate_gradient_type);
          },
          [](CylinderEstimationSolver &instance, const std::string &arg) {
            ceres::StringToNonlinearConjugateGradientType(
                arg, &instance.options_.nonlinear_conjugate_gradient_type);
          })

      // .def_property(
      //     "trust_region_minimizer_iterations_to_dump",
      //     [](CylinderEstimationSolver &instance) {
      //       return
      //       instance.options_.trust_region_minimizer_iterations_to_dump;
      //     },
      //     [](CylinderEstimationSolver &instance, const std::vector<int>& arg)
      //     {
      //       std::copy(arg.begin(), arg.end(),
      //                 instance.options_
      //                     .trust_region_minimizer_iterations_to_dump.begin());
      //     })

      ;

  return m.ptr();
}

} // namespace game
