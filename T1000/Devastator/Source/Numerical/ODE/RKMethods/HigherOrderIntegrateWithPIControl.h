#ifndef NUMERICAL_ODE_RK_METHODS_HIGHER_ORDER_INTEGRATE_WITH_PI_CONTROL_H
#define NUMERICAL_ODE_RK_METHODS_HIGHER_ORDER_INTEGRATE_WITH_PI_CONTROL_H

#include "CalculateError.h"
#include "CalculateNewY.h"
#include "ComputePIStepSize.h"
#include "HigherOrderStepWithPIControl.h"
#include "IntegrationInputs.h"
#include "StepInputs.h"

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

template <
  std::size_t S,
  std::size_t BHHSize,
  typename DerivativeType,
  typename ContainerT,
  typename Field = double>
class HigherOrderIntegrateWithPIControl
{
  public:

    static constexpr std::size_t default_max_steps_ {50000};

    HigherOrderIntegrateWithPIControl(
      CalculateNewY<S, DerivativeType, Field>& new_y,
      CalculateError<S, BHHSize, ContainerT, Field>& error,
      const ComputePIStepSize<Field>& pi_step,
      const std::size_t max_steps = default_max_steps_
      ):
      step_{new_y, error, pi_step},
      max_steps_{max_steps}
    {}

    HigherOrderIntegrateWithPIControl(
      CalculateNewY<S, DerivativeType, Field>&& new_y,
      CalculateError<S, BHHSize, ContainerT, Field>&& error,
      const ComputePIStepSize<Field>&& pi_step,
      const std::size_t max_steps = default_max_steps_
      ):
      step_{new_y, error, pi_step},
      max_steps_{max_steps}
    {}

    template <std::size_t N>
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

        h_used_save.emplace_back(step_.template step<N>(
          step_inputs,
          max_step_iterations));

        x_save.emplace_back(step_inputs.x_n_);
        y_save.emplace_back(step_inputs.y_n_);

        ++counter;
      }

      return std::make_tuple(x_save, y_save, h_used_save);
    }

  private:
    
    HigherOrderStepWithPIControl<S, BHHSize, DerivativeType, ContainerT, Field>
      step_;

    std::size_t max_steps_;
};

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_HIGHER_ORDER_INTEGRATE_WITH_PI_CONTROL_H
