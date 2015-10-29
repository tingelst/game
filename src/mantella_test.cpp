//
// Created by lars on 23.09.15.
//

#include <iostream>
#include <mantella>

#include <hep/ga.hpp>
#include "game/types.h"

// Armadillo
//#include <armadillo>

// Mantella
#include <mantella_bits/config.hpp>
#include <mantella_bits/optimisationProblem/blackBoxOptimisationBenchmark.hpp>

/*
namespace mant {
  namespace bbob {
    class SphereFunction : public BlackBoxOptimisationBenchmark {
     public:
      explicit SphereFunction(
          const arma::uword numberOfDimensions);

      std::string toString() const override;
#if defined(SUPPORT_MPI)
      std::vector<double> serialise() const;
      void deserialise(
          std::vector<double> serialisedOptimisationProblem);
#endif

     protected:
      double getObjectiveValueImplementation(
          const arma::Col<double>& parameter) const override;
    };
  }
}
 */

namespace game {
namespace pso {

using Rotor = game::cga::Rotor<double>;
using Vector = game::cga::EuclideanVector<double>;

class RotorEstimation : public mant::bbob::BlackBoxOptimisationBenchmark {
 public:
  explicit RotorEstimation(const arma::uword numberOfDimension)
      : mant::bbob::BlackBoxOptimisationBenchmark(numberOfDimension) {
    setParameterTranslation(getRandomParameterTranslation());
  }

  double getObjectiveValueImplementation(
      const arma::Col<double>& parameter) const {
    auto a = Vector{1.0, 0.0, 0.0};
    auto b = Vector{0.0, 1.0, 0.0};

    auto scale =
        1.0 / sqrt(parameter[0] * parameter[0] + parameter[1] * parameter[1] +
                   parameter[2] * parameter[2]);

    auto r = Rotor{scale * parameter[0], scale * parameter[1],
                   scale * parameter[2], scale * parameter[3]};

    Vector c = hep::grade<1>(r * a * ~r - b);

//    std::cout << c[0] << " ";
//    std::cout << c[1] << " ";
//    std::cout << c[2] << std::endl;


    return sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
  }

  std::string toString() const { return "rotor_estimation"; }
};

const int kNumPoints = 5;
const double points[] = {
    1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0,
};
class SphereFit : public mant::bbob::BlackBoxOptimisationBenchmark {
 public:
  explicit SphereFit(const arma::uword numberOfDimensions)
      : mant::bbob::BlackBoxOptimisationBenchmark(numberOfDimensions) {
    setParameterTranslation(getRandomParameterTranslation());
  }

  double getObjectiveValueImplementation(
      const arma::Col<double>& parameter) const {
    using Algebra = hep::algebra<double, 4, 1>;

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
    Infty ni{1.0, 1.0};
    Orig no{-0.5, 0.5};

    // Euclidean basis
    E1 e1{1.0};
    E2 e2{1.0};
    E3 e3{1.0};
    Scalar half{0.5};

    double residual = 0;

    // Create conformal sphere
    Sphere sphere =
        hep::eval(parameter[0] * e1 + parameter[1] * e2 + parameter[2] * e3 +
                  parameter[3] * ni + parameter[4] * no);

    //    std::cout << sphere[0] << " ";
    //    std::cout << sphere[1] << " ";
    //    std::cout << sphere[2] << " ";
    //    std::cout << sphere[3] << " ";
    //    std::cout << sphere[4] << std::endl;

    for (int i = 0; i < kNumPoints; ++i) {
      PointE3 euc_point{points[3 * i], points[3 * i + 1], points[3 * i + 2]};

      //      std::cout << euc_point[0] << " " << euc_point[1] << " " <<
      //      euc_point[2] << std::endl;

      Point point =
          hep::grade<1>(euc_point + half * euc_point * euc_point * ni + no);
      auto distance = hep::eval(hep::inner_prod(point, sphere))[0];
      residual += std::pow(distance, 2);
    }

    return residual;
  }

  std::string toString() const { return "pso_sphere_fit"; }
};

}  // namespace pso
}  // namespace game

int main() {
  // 1. Setup the optimisation problem.
//  std::shared_ptr<mant::OptimisationProblem> optimisationProblem(
//      new game::pso::SphereFit(5));

  std::shared_ptr<mant::OptimisationProblem> optimisationProblem(new game::pso::RotorEstimation(4));

//  auto optimisationAlgorithm =
//      mant::ParticleSwarmOptimisation(optimisationProblem, 10000);
//  optimisationAlgorithm.setMaximalNumberOfIterations(1000);
//  optimisationAlgorithm.optimise();

//    mant::HookeJeevesAlgorithm optimisationAlgorithm(optimisationProblem);
//    optimisationAlgorithm.setMaximalNumberOfIterations(100);
//    optimisationAlgorithm.optimise();

  mant::SimulatedAnnealing optimisationAlgorithm(optimisationProblem);
  optimisationAlgorithm.setMaximalNumberOfIterations(100000);
  optimisationAlgorithm.optimise();


  // 3. Get your result!

  std::cout << "isFinished: " << optimisationAlgorithm.isFinished() << '\n';
  std::cout << "isTerminated: " << optimisationAlgorithm.isTerminated() << '\n';
  std::cout << "numberOfIterations: "
            << optimisationAlgorithm.getNumberOfIterations() << '\n';
  std::cout << "bestObjectiveValue: "
            << optimisationAlgorithm.getBestObjectiveValue() << '\n';
  std::cout << "bestParameter: " << arma::normalise(optimisationAlgorithm.getBestParameter())
            << std::endl;


  return 0;
}