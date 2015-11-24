#ifndef GAME_GAME_ADEPT_AUTODIFF_COST_FUNCTION_H_
#define GAME_GAME_ADEPT_AUTODIFF_COST_FUNCTION_H_

#include "ceres/internal/autodiff.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/sized_cost_function.h"
#include "ceres/types.h"
#include "glog/logging.h"

namespace game {

template <typename CostFunctor,
          int kNumResiduals,  // Number of residuals, or ceres::DYNAMIC.
          int N0,             // Number of parameters in block 0.
          int N1 = 0,         // Number of parameters in block 1.
          int N2 = 0,         // Number of parameters in block 2.
          int N3 = 0,         // Number of parameters in block 3.
          int N4 = 0,         // Number of parameters in block 4.
          int N5 = 0,         // Number of parameters in block 5.
          int N6 = 0,         // Number of parameters in block 6.
          int N7 = 0,         // Number of parameters in block 7.
          int N8 = 0,         // Number of parameters in block 8.
          int N9 = 0>         // Number of parameters in block 9.
class AdeptAutoDiffCostFunction
    : public ceres::SizedCostFunction<kNumResiduals, N0, N1, N2, N3, N4, N5, N6,
                                      N7, N8, N9> {
public:
  // Takes ownership of functor. Uses the template-provided value for the
  // number of residuals ("kNumResiduals").
  explicit AdeptAutoDiffCostFunction(CostFunctor* functor) : functor_(functor) {
    CHECK_NE(kNumResiduals, ceres::DYNAMIC)
        << "Can't run the fixed-size constructor if the "
        << "number of residuals is set to ceres::DYNAMIC.";
  }

  // Takes ownership of functor. Ignores the template-provided
  // kNumResiduals in favor of the "num_residuals" argument provided.
  //
  // This allows for having autodiff cost functions which return varying
  // numbers of residuals at runtime.
  AdeptAutoDiffCostFunction(CostFunctor* functor, int num_residuals)
      : functor_(functor) {
    CHECK_EQ(kNumResiduals, ceres::DYNAMIC)
        << "Can't run the dynamic-size constructor if the "
        << "number of residuals is not ceres::DYNAMIC.";
    ceres::SizedCostFunction<kNumResiduals, N0, N1, N2, N3, N4, N5, N6, N7, N8,
                             N9>::set_num_residuals(num_residuals);
  }

  virtual ~AdeptAutoDiffCostFunction() {}

  virtual bool Evaluate(double const* const* parameters, double* residuals,
                        double** jacobians) const {
    if (!jacobians) {
      return ceres::internal::VariadicEvaluate<CostFunctor, double, N0, N1, N2,
                                               N3, N4, N5, N6, N7, N8,
                                               N9>::Call(*functor_, parameters,
                                                         residuals);
    }

    adept::Stack stack;

    adept::adouble ax[N0];
    adept::set_values(&ax[0], N0, parameters[0]);

    stack.new_recording();
    adept::adouble aresiduals[kNumResiduals];

    (*functor_)(ax, aresiduals);

    for (int i = 0; i < kNumResiduals; ++i) {
      residuals[i] = adept::value(aresiduals[i]);
    }

    stack.independent(&ax[0], N0);
    stack.dependent(&aresiduals[0], kNumResiduals);

    //stack.jacobian(jacobians[0], true);
    stack.jacobian_reverse(jacobians[0], true);
    //stack.jacobian_forward(jacobians[0], true);

    return true;
  }

private:
  ceres::internal::scoped_ptr<CostFunctor> functor_;
};
}  // namespace game
#endif  // GAME_GAME_ADEPT_AUTODIFF_COST_FUNCTION_H_
