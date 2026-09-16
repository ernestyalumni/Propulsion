#ifndef NUMERICAL_ODE_RK_METHODS_STEP_WITH_PI_CONTROL_H
#define NUMERICAL_ODE_RK_METHODS_STEP_WITH_PI_CONTROL_H

#include "CalculateNewYAndError.h"
#include "CalculateScaledError.h"
#include "ComputePIStepSize.h"
#include "PIStepSizeControl.h"
#include "StepInputs.h"

#include <cmath>
#include <cstdint>
#include <utility>
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
      new_y_and_err_{std::move(new_y_and_err)},
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
      static_assert(N > 0, "An ODE state must have positive dimension");
      if (inputs.y_n_.size() != N || inputs.dydx_n_.size() != N)
        throw std::invalid_argument("ODE state dimension mismatch");
      if (!std::isfinite(inputs.x_n_) || !std::isfinite(inputs.h_n_) ||
          inputs.h_n_ == Field(0))
        throw std::invalid_argument("ODE requires finite time and nonzero finite step");
      for (std::size_t i=0; i<N; ++i)
        if (!std::isfinite(inputs.y_n_[i]) || !std::isfinite(inputs.dydx_n_[i]))
          throw std::domain_error("Nonfinite ODE state or derivative");
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
        if (!std::isfinite(h) || !std::isfinite(inputs.x_n_ + h) ||
            inputs.x_n_ + h == inputs.x_n_)
          throw std::runtime_error("ODE step cannot advance finite time");

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

        if (!std::isfinite(error))
          throw std::domain_error("Nonfinite ODE error estimate");
        for (std::size_t i=0; i<N; ++i)
          if (!std::isfinite(y_out[i]) ||
              !std::isfinite(inputs.k_coefficients_.get_ith_coefficient(S)[i]))
            throw std::domain_error("Nonfinite ODE trial state or final derivative");

        h_np1 = pi_step_.compute_new_step_size(
          error,
          pi_control_.get_previous_error(),
          h,
          pi_control_.get_is_rejected());

        pi_control_.accept_computed_step(error);

        ++iterations;
      }

      if (!(error <= Field(1)))
      {
        throw std::runtime_error("Iterations exceeded max. iterations");
      }

      inputs.y_n_ = y_out;
      inputs.dydx_n_ = inputs.k_coefficients_.get_ith_coefficient(S);
      inputs.h_n_ = h_np1;
      inputs.x_n_ += h;

      return h;
    }

    // A new IVP must not inherit the preceding integration's PI history.
    void reset() { pi_control_ = PIStepSizeControl<Field>{}; }

    template <typename ContainerT>
    ContainerT calculate_derivative(
      const Field x,
      const ContainerT& y)
    {
      return new_y_and_err_.calculate_derivative(x, y);
    }

  private:

    CalculateNewYAndError<S, DerivativeType, Field> new_y_and_err_;
    CalculateScaledError<Field> scaled_error_;
    ComputePIStepSize<Field> pi_step_;
    PIStepSizeControl<Field> pi_control_;
};

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_STEP_WITH_PI_CONTROL_H
