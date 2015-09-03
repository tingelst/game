// sphere_fit.cpp
//
// Example from Foundations of Geometric Algebra Computing
// by Dietmar Hildenbrand
//
// Page 68.

#include <fstream>
#include <iostream>

#include <hep/ga.hpp>
#include <ceres/ceres.h>

const int kNumPoints = 5;
const double points[] = {
    1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0,
};

struct SphereFitCostFunction {
  SphereFitCostFunction(const double* point) : point_(point) {}

  template <typename T>
  bool operator()(const T* const s /* sphere: 5 parameters */,
                  T* residual /* 1 parameter */) const {
    // Types
    using Algebra = hep::algebra<T, 4, 1>;
    using Scalar = hep::multi_vector<Algebra, hep::list<0>>;
    using Infty = hep::multi_vector<Algebra, hep::list<8, 16>>;
    using Orig = hep::multi_vector<Algebra, hep::list<8, 16>>;
    using Point = hep::multi_vector<Algebra, hep::list<1, 2, 4, 8, 16>>;
    using Sphere = hep::multi_vector<Algebra, hep::list<1, 2, 4, 8, 16>>;
    using PointE3 = hep::multi_vector<Algebra, hep::list<1, 2, 4>>;
    using E1 = hep::multi_vector<Algebra, hep::list<1>>;
    using E2 = hep::multi_vector<Algebra, hep::list<2>>;
    using E3 = hep::multi_vector<Algebra, hep::list<4>>;

    // Conformal split
    Infty ni{static_cast<T>(1.0), static_cast<T>(1.0)};
    Orig no{static_cast<T>(-0.5), static_cast<T>(0.5)};

    // Euclidean basis
    E1 e1{static_cast<T>(1.0)};
    E2 e2{static_cast<T>(1.0)};
    E3 e3{static_cast<T>(1.0)};

    Scalar half{static_cast<T>(0.5)};

    // Create Euclidean point (vector)
    PointE3 euc_point{static_cast<T>(point_[0]), static_cast<T>(point_[1]),
                      static_cast<T>(point_[2])};

    // Create conformal point
    Point point =
        hep::grade<1>(euc_point + half * euc_point * euc_point * ni + no);

    // Create conformal sphere
    Sphere sphere =
        hep::eval(static_cast<T>(s[0]) * e1 + static_cast<T>(s[1]) * e2 +
                  static_cast<T>(s[2]) * e3 + static_cast<T>(s[3]) * ni +
                  static_cast<T>(s[4]) * no);

    // Evaluate distance
    auto distance = hep::eval(hep::inner_prod(point, sphere));

    residual[0] = distance[0];

    return true;
  }

  static ceres::CostFunction* Create(const double* point) {
    return (new ceres::AutoDiffCostFunction<SphereFitCostFunction, 1, 5>(
        new SphereFitCostFunction(point)));
  }

 private:
  const double* point_;
};

class LoggingCallback : public ceres::IterationCallback {
 public:
  explicit LoggingCallback(bool log_to_stdout)
      : log_to_stdout_(log_to_stdout) {}

  ~LoggingCallback() {}

  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
    if (log_to_stdout_) {
      std::cout << summary.iteration << std::endl;
    }
    return ceres::SOLVER_CONTINUE;
  }

 private:
  const bool log_to_stdout_;
};

class LoggingCallback2 : public ceres::IterationCallback {
 public:
  explicit LoggingCallback2(bool log_to_stdout)
      : log_to_stdout_(log_to_stdout) {}

  ~LoggingCallback2() {}

  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
    if (log_to_stdout_) {
      std::cout << "Lars" << std::endl;
    }
    return ceres::SOLVER_CONTINUE;
  }

 private:
  const bool log_to_stdout_;
};

bool DumpSummaryToFile(const std::string& fname,
                       const ceres::Solver::Summary& summary);

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  // Initial values of spheres in basis {e1, e2, e3, ni, no}
  //  double sphere[] = {-103.0, 100000.0, 231.0, -939371.0, 541.0};
  double sphere[] = {1.0, 1.0, 1.0, 1.0, 1.0};

  ceres::Problem problem;
  for (int i = 0; i < kNumPoints; ++i) {
    ceres::CostFunction* cost_function =
        SphereFitCostFunction::Create(&points[3 * i]);
    problem.AddResidualBlock(cost_function, NULL /* squared loss */, sphere);
  }

  auto log_cb = std::unique_ptr<LoggingCallback>(new LoggingCallback(true));
  auto log_cb2 = std::unique_ptr<LoggingCallback2>(new LoggingCallback2(true));

  ceres::Solver::Options options;
  options.max_num_iterations = 25;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  options.trust_region_problem_dump_directory =
      "/home/lars/devel/game_ws/dump/sphere_fit";
  options.trust_region_minimizer_iterations_to_dump =
      std::vector<int>{1, 2, 3, 4};

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << "\n";

  std::cout << "Initial sphere: {1.0,1.0,1.0,1.0,1.0}" << std::endl;

  std::cout << "Final sphere: {" << sphere[0] << "," << sphere[1] << ","
            << sphere[2] << "," << sphere[3] << "," << sphere[4] << "}"
            << std::endl;

  std::cout << "Scaled final sphere: {" << sphere[0] / sphere[4] << ","
            << sphere[1] / sphere[4] << "," << sphere[2] / sphere[4] << ","
            << sphere[3] / sphere[4] << "," << sphere[4] / sphere[4] << "}"
            << std::endl;

  std::string filename{"/home/lars/devel/game_ws/dump/sphere_fit/sphere_fit_summary.m"};
  DumpSummaryToFile(filename, summary);

  return 0;
}

bool DumpSummaryToFile(const std::string& filename,
                       const ceres::Solver::Summary& summary) {
  std::ofstream outf(filename);
  if (outf) {
    outf << "function summary = load_summary()" << "\n";
    outf << "summary.brief_report = \'" << summary.BriefReport() << "\'\n";
    outf << "summary.num_parameter_blocks = " << summary.num_parameter_blocks << "\n";
    outf << "summary.num_parameters = " << summary.num_parameters << "\n";
    outf << "summary.num_residual_blocks = " << summary.num_residual_blocks << "\n";
    outf << "summary.num_residuals = " << summary.num_residuals << "\n";
    auto its = summary.iterations;
    for (int i = 0; i < its.size(); ++i)
    {
      outf << "summary.iterations(" << i+1 << ").iteration = " << its[i].iteration << "\n";
      outf << "summary.iterations(" << i+1 << ").cost = " << its[i].cost << "\n";
      outf << "summary.iterations(" << i+1 << ").cost_change = " << its[i].cost_change << "\n";
      outf << "summary.iterations(" << i+1 << ").gradient_max_norm = " << its[i].gradient_max_norm << "\n";
      outf << "summary.iterations(" << i+1 << ").step_norm = " << its[i].step_norm << "\n";
      outf << "summary.iterations(" << i+1 << ").relative_decrease = " << its[i].relative_decrease << "\n";
      outf << "summary.iterations(" << i+1 << ").trust_region_radius = " << its[i].trust_region_radius << "\n";
      outf << "summary.iterations(" << i+1 << ").eta = " << its[i].eta << "\n";
      outf << "summary.iterations(" << i+1 << ").linear_solver_iterations = " << its[i].linear_solver_iterations << "\n";
    }
  }
}
