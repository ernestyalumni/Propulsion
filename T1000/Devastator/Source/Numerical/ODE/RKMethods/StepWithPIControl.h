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
      const CalculateNewYAndError<S, DerivativeType, Field>& new_y_and_err,
      const CalculateScaledError<Field>& scaled_error,
      const ComputePIStepSize<Field>& pi_step
      ):
      new_y_and_err_{new_y_and_err},
      scaled_error_{scaled_error},
      pi_step_{pi_step},
      pi_control_{}
    {}

    template <std::size_t N, typename ContainerT>
    void step(StepInputs& inputs, const std::size_t max_iterations = 100)
    {
      const std::size_t iterations {0};
      ContainerT y_out;
      Field h {inputs.h_n_};
      Field error {1.0};

      while (error >= 1.0 && iterations < max_iterations)
      {
        y_out = new_y_and_err_.calculate_new_y(
          h,
          inputs.x_n_,
          inputs.y_n_,
          inputs.k_coefficients_.get_ith_coefficient(S),
          inputs.k_coefficients_);

        auto calculated_error = new_y_and_err_.calculate_error(
          h,
          inputs.k_coefficients_);

        auto error = scaled_error_.operator()<ContainerT, N>(
          inputs.y_n_,
          y_out,
          calculated_error);

        h = pi_step_.compute_new_step_size(
          error,
          pi_control_.get_previous_error(),
          h,
          pi_control_.get_is_rejected())

        pi_control_.accept_computed_step(error);

        ++iterations;
      }

      if (iterations >= max_iterations)
      {
        throw std::runtime_error("Iterations exceeded max. iterations");
      }

      inputs.y_n_ = y_out;
      inputs.h_n_ = h;
      inputs.x_n_ += h;
    }

  private:

    CalculateNewYAndError<S, DerivativeType, Field>& new_y_and_err_;
    CalculateScaledError<Field> scaled_error_;
    ComputePIStepSize<Field>& pi_step_;
    PIStepSizeControl<Field>& pi_control_;
};

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_STEP_WITH_PI_CONTROL_H
