#ifndef NUMERICAL_ODE_RK_METHODS_COMPUTE_PI_STEP_SIZE_H
#define NUMERICAL_ODE_RK_METHODS_COMPUTE_PI_STEP_SIZE_H

#include <algorithm>
#include <cassert>
#include <cmath>

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{

template <typename Field = double>
class ComputePIStepSize
{
  public:

    static constexpr Field max_scale_after_rejection {static_cast<Field>(1)};

    ComputePIStepSize(
      const Field alpha,
      const Field beta,
      const Field min_scale = 0.2,
      const Field max_scale = 5.0,
      const Field safety_factor = 0.9
      ):
      alpha_{alpha},
      beta_{beta},
      min_scale_{min_scale},
      max_scale_{max_scale},
      safety_factor_{safety_factor}
    {
      assert(alpha >= static_cast<Field>(0));
      assert(beta >= static_cast<Field>(0));
      assert(safety_factor >= static_cast<Field>(0));
    }

    virtual ~ComputePIStepSize() = default;

    Field compute_new_step_size(
      const Field error,
      const Field previous_error,
      const Field h,
      const bool was_rejected) const
    {
      Field scale {static_cast<Field>(0)};

      if (error <= static_cast<Field>(1))
      {
        // If there's no error, we can step forward by the largest amount.
        scale = (error == static_cast<Field>(0)) ? max_scale_ :
          // Includes Lund-stabilization with the multiplication of the factor
          // with previous error.
          safety_factor_ * std::pow(error, -alpha_) * std::pow(
            previous_error,
            beta_);

        // Ensure min_scale_ <= h_new / h <= max_scale_.
        scale = std::min(std::max(scale, min_scale_), max_scale_);

        // Don't let the step h increase if previous one was rejected.
        return was_rejected ? h * std::min(scale, static_cast<Field>(1)) :
          h * scale;
      }

      assert(error > static_cast<Field>(1));

      scale = std::max(safety_factor_ * std::pow(error, -alpha_), min_scale_);
      return h * scale;
    }

  private:

    Field alpha_;
    Field beta_;

    // Hairer, Norsett, and Wanner (1993), Ordinary Differential Equations, Vol.
    // 1, pp. 168, calls these 2 factors facmin, facmax, respectively.
    // fac could stand for factor.

    Field min_scale_;

    // From pp. 168 of Hairer, Norsett, and Wanner (1993), the maximal step size
    // increase, facmax, usually chosen between 1.5 and 5, prevents code from
    // too large step increases and contributes to its safely.
    // It's also advisable to put facmax = 1 in steps right after a
    // step-rejection.
    Field max_scale_;

    Field safety_factor_;
};

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_COMPUTE_PI_STEP_SIZE_H
