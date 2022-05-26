#ifndef NUMERICAL_ODE_RK_METHODS_STEP_WITH_PI_CONTROL_H
#define NUMERICAL_ODE_RK_METHODS_STEP_WITH_PI_CONTROL_H

#include "CalculateNewYAndError.h"
#include "CalculateScaledError.h"
#include "ComputePIStepSize.h"
#include "PIStepSizeControl.h"
#include "StepInputs.h"

#include <stdexcept>

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{

template <std::size_t S, typename DerivativeType, typename Field = double>
class StepWithPIControl
{
  public:

    StepWithPIControl(
      CalculateNewYAndError<S, DerivativeType, Field>& new_y_and_err,
      const CalculateScaledError<Field>& scaled_error,
      const ComputePIStepSize<Field>& pi_step
      ):
      new_y_and_err_{new_y_and_err},
      scaled_error_{scaled_error},
      pi_step_{pi_step},
      pi_control_{}
    {}

    StepWithPIControl(
      CalculateNewYAndError<S, DerivativeType, Field>&& new_y_and_err,
      const CalculateScaledError<Field>&& scaled_error,
      const ComputePIStepSize<Field>&& pi_step
      ):
      new_y_and_err_{new_y_and_err},
      scaled_error_{scaled_error},
      pi_step_{pi_step},
      pi_control_{}
    {}

    //--------------------------------------------------------------------------
    /// \return h, the step value used to compute the new y with, *not* the h
    /// value computed for the next step.
    //--------------------------------------------------------------------------
    template <std::size_t N, typename ContainerT>
    Field step(
      StepInputs<S, ContainerT, Field>& inputs,
      const std::size_t max_iterations = 100)
    {
      std::size_t iterations {0};
      ContainerT y_out;
      Field h {inputs.h_n_};
      Field h_np1 {inputs.h_n_};
      Field error {1.1};

      while (error > 1.0 && iterations < max_iterations)
      {
        // If this step is repeated at least once, then we use the newly
        // computed step to calculate the new y, y_{n + 1}, with.
        h = h_np1;

        y_out = new_y_and_err_.calculate_new_y(
          h,
          inputs.x_n_,
          inputs.y_n_,
          inputs.dydx_n_,
          inputs.k_coefficients_);

        auto calculated_error = new_y_and_err_.calculate_error(
          h,
          inputs.k_coefficients_);

        error = scaled_error_.template operator()<ContainerT, N>(
          inputs.y_n_,
          y_out,
          calculated_error);

        h_np1 = pi_step_.compute_new_step_size(
          error,
          pi_control_.get_previous_error(),
          h,
          pi_control_.get_is_rejected());

        pi_control_.accept_computed_step(error);

        ++iterations;
      }

      if (iterations >= max_iterations)
      {
        throw std::runtime_error("Iterations exceeded max. iterations");
      }

      inputs.y_n_ = y_out;
      inputs.dydx_n_ = inputs.k_coefficients_.get_ith_coefficient(S);
      inputs.h_n_ = h_np1;
      inputs.x_n_ += h;

      return h;
    }

  private:

    CalculateNewYAndError<S, DerivativeType, Field>& new_y_and_err_;
    CalculateScaledError<Field> scaled_error_;
    ComputePIStepSize<Field> pi_step_;
    PIStepSizeControl<Field> pi_control_;
};

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_STEP_WITH_PI_CONTROL_H
