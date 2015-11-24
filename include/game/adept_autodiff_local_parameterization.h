#ifndef GAME_GAME_ADEPT_AUTODIFF_LOCAL_PARAMETERIZATION_H_
#define GAME_GAME_ADEPT_AUTODIFF_LOCAL_PARAMETERIZATION_H_

#include "ceres/local_parameterization.h"
#include "ceres/internal/autodiff.h"
#include "ceres/internal/scoped_ptr.h"

#include <adept.h>

namespace game {

template <typename Functor, int kGlobalSize, int kLocalSize>
class AdeptAutoDiffLocalParameterization : public ceres::LocalParameterization {
public:
  AdeptAutoDiffLocalParameterization() : functor_(new Functor()) {}

  // Takes ownership of functor.
  explicit AdeptAutoDiffLocalParameterization(Functor* functor)
      : functor_(functor) {}

  virtual ~AdeptAutoDiffLocalParameterization() {}
  virtual bool Plus(const double* x, const double* delta,
                    double* x_plus_delta) const {
    return (*functor_)(x, delta, x_plus_delta);
  }
  //
  virtual bool ComputeJacobian(const double* x, double* jacobian) const {
    double zero_delta[kLocalSize];
    for (int i = 0; i < kLocalSize; ++i) {
      zero_delta[i] = 0.0;
    }

    double x_plus_delta[kGlobalSize];
    for (int i = 0; i < kGlobalSize; ++i) {
      x_plus_delta[i] = 0.0;
    }

    adept::Stack stack;
    adept::adouble ax[kGlobalSize];
    adept::adouble adelta[kLocalSize];
    adept::set_values(&ax[0], kGlobalSize, x);
    adept::set_values(&adelta[0], kLocalSize, zero_delta);

    stack.new_recording();
    adept::adouble ax_plus_delta[kGlobalSize];

    (*functor_)(ax, adelta, ax_plus_delta);

    stack.independent(&adelta[0], kLocalSize);
    stack.dependent(&ax_plus_delta[0], kGlobalSize);

    stack.jacobian(jacobian, true);
    //stack.jacobian_reverse(jacobian, true);

    return true;
  }

  virtual int GlobalSize() const { return kGlobalSize; }
  virtual int LocalSize() const { return kLocalSize; }

private:
  ceres::internal::scoped_ptr<Functor> functor_;
};
}

#endif  //  GAME_GAME_ADEPT_AUTODIFF_LOCAL_PARAMETERIZATION_H_
