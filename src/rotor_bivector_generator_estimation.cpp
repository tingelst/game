#include <iostream>
#include <hep/ga.hpp>

#include <ceres/ceres.h>
#include <glog/logging.h>

const int kNumPoints = 10;
const double a_data[] = {-0.1022, -0.8637, -0.2414, 0.0774,  0.3192,
                         -1.2141, 0.3129,  -1.1135, -0.8649, -0.0068,
                         -0.0301, 1.5326,  -0.1649, -0.7697, 0.6277,
                         0.3714,  1.0933,  -0.2256, 1.1093,  1.1174};

const double b_data[] = {-0.7991, -0.3433, -0.0537, 0.2478,  -0.8919,
                         -0.8835, -0.8079, -0.8277, -0.4384, 0.7456,
                         1.3123,  0.7923,  -0.7490, -0.2420, 0.6355,
                         -0.3579, 0.3513,  -1.0596, 1.5223,  -0.4020};

template <typename T>
void Exp(const T* bivector, T* rotor) {
  T psi = sqrt(bivector[0] * bivector[0] + bivector[1] * bivector[1] +
               bivector[2] * bivector[2]);
  rotor[0] = cos(psi);
  rotor[1] = -sin(psi) * (bivector[0] / psi);
  rotor[2] = -sin(psi) * (bivector[1] / psi);
  rotor[3] = -sin(psi) * (bivector[2] / psi);
}

struct DistanceErrorCostFunctor {
  DistanceErrorCostFunctor(const double* a, const double* b) : a_(a), b_(b) {}

  template <typename T>
  bool operator()(const T* const bivector, T* residuals) const {
    using Algebra = hep::algebra<T, 3, 0>;
    using Scalar = hep::multi_vector<Algebra, hep::list<0> >;
    using Rotor = hep::multi_vector<Algebra, hep::list<0, 3, 5, 6> >;
    using Point = hep::multi_vector<Algebra, hep::list<1, 2, 4> >;
    using Vector = hep::multi_vector<Algebra, hep::list<1, 2, 4> >;

    Point a{static_cast<T>(a_[0]), static_cast<T>(a_[1]), static_cast<T>(0.0)};
    Point b{static_cast<T>(b_[0]), static_cast<T>(b_[1]), static_cast<T>(0.0)};

    Rotor rot;

    Exp(bivector, &rot[0]);

    Point rar = hep::grade<1>(rot * a * ~rot);

    Vector dist = rar - b;

    residuals[0] = dist[0];
    residuals[1] = dist[1];
    residuals[2] = dist[2];

    return true;
  }

  static ceres::CostFunction* Create(const double* a, const double* b) {
    return (new ceres::AutoDiffCostFunction<DistanceErrorCostFunctor, 3, 3>(
        new DistanceErrorCostFunctor(a, b)));
  }

 private:
  const double* a_;
  const double* b_;
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  double B[] = {0.0, 1.0, 0};  // {e12,e13,e23}

  ceres::Problem problem;
  for (int i = 0; i < kNumPoints; ++i) {
    ceres::CostFunction* cost_function =
        DistanceErrorCostFunctor::Create(&a_data[2 * i], &b_data[2 * i]);
    problem.AddResidualBlock(cost_function, NULL, B);
  }

  ceres::Solver::Options options;
  options.max_num_iterations = 10;
  options.linear_solver_type = ceres::DENSE_QR;
  options.num_threads = 12;
  options.num_linear_solver_threads = 12;
  options.minimizer_progress_to_stdout = true;
  options.function_tolerance = 10e-12;
  options.parameter_tolerance = 10e-12;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  double rotor[4];

  Exp(&B[0], &rotor[0]);

  std::cout << "[ " << B[0] << " " << B[1] << " " << B[2] << " ]" << std::endl;

  std::cout << "Final rotor: {" << rotor[0] << "," << rotor[1] << ","
            << rotor[2] << "," << rotor[3] << "}\n";

  return 0;
}
