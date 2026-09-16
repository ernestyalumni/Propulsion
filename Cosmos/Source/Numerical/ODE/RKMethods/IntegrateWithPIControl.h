#ifndef NUMERICAL_ODE_RK_METHODS_INTEGRATE_WITH_PI_CONTROL_H
#define NUMERICAL_ODE_RK_METHODS_INTEGRATE_WITH_PI_CONTROL_H

#include "CalculateNewYAndError.h"
#include "CalculateScaledError.h"
#include "ComputePIStepSize.h"
#include "IntegrationInputs.h"
#include "StepInputs.h"
#include "StepWithPIControl.h"
#include "calculate_hermite_interpolation.h"

#include <cassert>
#include <cstdint>
#include <tuple>
#include <vector>

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{

template <std::size_t S, typename DerivativeType, typename Field = double>
class IntegrateWithPIControl
{
  public:

    static constexpr std::size_t default_max_steps_ {50000};

    IntegrateWithPIControl(
      CalculateNewYAndError<S, DerivativeType, Field>& new_y_and_err,
      const CalculateScaledError<Field>& scaled_error,
      const ComputePIStepSize<Field>& pi_step,
      const std::size_t max_steps = default_max_steps_
      ):
      step_{new_y_and_err, scaled_error, pi_step},
      max_steps_{max_steps}
    {}

    IntegrateWithPIControl(
      CalculateNewYAndError<S, DerivativeType, Field>&& new_y_and_err,
      const CalculateScaledError<Field>&& scaled_error,
      const ComputePIStepSize<Field>&& pi_step,
      const std::size_t max_steps = default_max_steps_
      ):
      step_{std::move(new_y_and_err), std::move(scaled_error), std::move(pi_step)},
      max_steps_{max_steps}
    {}

    // The observer receives accepted steps only: (time, state, accepted h).
    // Returns (final time, final state, accepted step count). No trajectory
    // allocation is required when the caller only wants the final state.
    template <std::size_t N, typename ContainerT, typename Observer>
    std::tuple<Field, ContainerT, std::size_t> integrate_with_observer(
      const IntegrationInputs<ContainerT, Field>& inputs,
      Observer observer, const std::size_t max_step_iterations = 100,
      const Field h_min = Field(0))
    {
      if (!std::isfinite(inputs.x_1_) || !std::isfinite(inputs.x_2_) ||
          !std::isfinite(inputs.h_0_) || !std::isfinite(h_min) || h_min < Field(0))
        throw std::invalid_argument("Invalid ODE interval or step size");
      if (inputs.y_0_.size() != N)
        throw std::invalid_argument("ODE state dimension mismatch");
      for (std::size_t i=0; i<N; ++i)
        if (!std::isfinite(inputs.y_0_[i]))
          throw std::domain_error("Nonfinite initial ODE state");
      step_.reset();
      if (inputs.x_1_ == inputs.x_2_)
        return {inputs.x_1_, inputs.y_0_, 0};
      if (inputs.h_0_ == Field(0))
        throw std::invalid_argument("Initial ODE step must be nonzero");
      const bool forward = inputs.x_2_ > inputs.x_1_;
      const Field h = forward ? std::abs(inputs.h_0_) : -std::abs(inputs.h_0_);
      StepInputs<S, ContainerT, Field> current {
        inputs.y_0_, step_.calculate_derivative(inputs.x_1_, inputs.y_0_),
        h, inputs.x_1_};
      for (std::size_t count=0; count<max_steps_; ++count)
      {
        const Field remaining = inputs.x_2_ - current.x_n_;
        if (!std::isfinite(remaining))
          throw std::invalid_argument("ODE interval is not representable");
        if (!std::isfinite(current.h_n_) || current.h_n_ == Field(0) ||
            (current.h_n_ > Field(0)) != forward)
          throw std::runtime_error("Invalid proposed ODE step");
        const Field next = current.x_n_ + current.h_n_;
        const bool last = std::abs(current.h_n_) >= std::abs(remaining) ||
          !std::isfinite(next) || (forward ? next >= inputs.x_2_ : next <= inputs.x_2_);
        if (last) current.h_n_ = remaining;
        else if (std::abs(current.h_n_) < h_min)
          throw std::runtime_error("ODE step below minimum");
        const Field used = step_.template step<N, ContainerT>(current, max_step_iterations);
        // A rejected endpoint trial may accept a smaller step. Only snap when
        // the full remaining interval was actually accepted.
        if (last && used == remaining) current.x_n_ = inputs.x_2_;
        observer(current.x_n_, current.y_n_, used);
        if (current.x_n_ == inputs.x_2_)
          return {current.x_n_, current.y_n_, count+1};
      }
      throw std::runtime_error("ODE integration exceeded maximum steps");
    }

    template <std::size_t N, typename ContainerT>
    std::tuple<std::vector<Field>, std::vector<ContainerT>, std::vector<Field>>
      integrate(const IntegrationInputs<ContainerT, Field>& inputs,
        const std::size_t max_step_iterations = 100)
    {
      std::vector<Field> times {inputs.x_1_};
      std::vector<ContainerT> states {inputs.y_0_};
      std::vector<Field> steps;
      integrate_with_observer<N>(inputs,
        [&](Field x, const ContainerT& y, Field h)
        {
          times.push_back(x); states.push_back(y); steps.push_back(h);
        }, max_step_iterations);
      return {std::move(times), std::move(states), std::move(steps)};
    }

    template <std::size_t N, typename ContainerT>
    std::tuple<std::vector<ContainerT>, std::vector<Field>>
      integrate_for_dense_output(
        const IntegrationInputsForDenseOutput<ContainerT, Field>& inputs,
        const std::size_t max_step_iterations = 100)
    {
      StepInputs<S, ContainerT, Field> step_inputs {
        inputs.y_0_,
        step_.calculate_derivative(inputs.x_1_, inputs.y_0_),
        inputs.h_,
        inputs.x_1_};

      std::vector<ContainerT> y_save {};
      y_save.reserve(inputs.x_.size());
      y_save.emplace_back(inputs.y_0_);
      std::vector<Field> h_used_save {};
      h_used_save.reserve(inputs.x_.size());

      Field x_n {inputs.x_1_};
      Field x_np1 {inputs.x_1_};
      ContainerT y_n {inputs.y_0_};
      ContainerT y_np1 {inputs.y_0_};
      ContainerT dydx_n {step_inputs.dydx_n_};
      ContainerT dydx_np1 {step_inputs.dydx_n_};

      for (auto iter = inputs.x_.begin() + 1; iter != inputs.x_.end(); ++iter)
      {
        Field h_used {};

        while (*iter > x_np1)
        {
          h_used = step_.template step<N, ContainerT>(
            step_inputs,
            max_step_iterations);
          x_n = x_np1;
          x_np1 = step_inputs.x_n_;
          y_n = y_np1;
          y_np1 = step_inputs.y_n_;
          dydx_n = dydx_np1;
          dydx_np1 = step_inputs.dydx_n_;
        }

        h_used_save.emplace_back(h_used);

        assert(*iter <= x_np1);

        const auto y_out = calculate_hermite_interpolation<ContainerT, Field>(
          y_n,
          y_np1,
          dydx_n,
          dydx_np1,
          (*iter - x_n) / (x_np1 - x_n),
          (x_np1 - x_n));

        y_save.emplace_back(y_out);
      }

      return std::make_tuple(y_save, h_used_save);
    }    

  private:
    
    StepWithPIControl<S, DerivativeType, Field> step_;

    std::size_t max_steps_;
};

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_INTEGRATE_WITH_PI_CONTROL_H
