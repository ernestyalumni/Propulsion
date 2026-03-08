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
      step_{new_y_and_err, scaled_error, pi_step},
      max_steps_{max_steps}
    {}

    template <std::size_t N, typename ContainerT>
    std::tuple<std::vector<Field>, std::vector<ContainerT>, std::vector<Field>>
      integrate(
        const IntegrationInputs<ContainerT, Field>& inputs,
        const std::size_t max_step_iterations = 100)
    {
      std::size_t counter {0};
      StepInputs<S, ContainerT, Field> step_inputs {
        inputs.y_0_,
        step_.calculate_derivative(inputs.x_1_, inputs.y_0_),
        inputs.h_0_,
        inputs.x_1_};

      std::vector<ContainerT> y_save {};
      y_save.reserve(max_steps_);
      std::vector<Field> x_save {};
      x_save.reserve(max_steps_);
      std::vector<Field> h_used_save {};

      x_save.emplace_back(inputs.x_1_);
      y_save.emplace_back(inputs.y_0_);

      while (counter < max_steps_)
      {
        // StepInputs instance gets mutated by step class method of
        // StepWithPIControls.
        if ((step_inputs.x_n_ - inputs.x_2_) * (inputs.x_2_ - inputs.x_1_) >
          static_cast<Field>(0))
        {
          return std::make_tuple(x_save, y_save, h_used_save);
        }

        h_used_save.emplace_back(step_.template step<N, ContainerT>(
          step_inputs,
          max_step_iterations));

        x_save.emplace_back(step_inputs.x_n_);
        y_save.emplace_back(step_inputs.y_n_);

        ++counter;
      }

      return std::make_tuple(x_save, y_save, h_used_save);
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
