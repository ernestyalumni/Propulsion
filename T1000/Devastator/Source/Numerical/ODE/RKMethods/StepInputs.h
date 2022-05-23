#ifndef NUMERICAL_ODE_RK_METHODS_STEP_INPUTS_H
#define NUMERICAL_ODE_RK_METHODS_STEP_INPUTS_H

#include "CalculateNewYAndError.h"
#include "CalculateScaledError.h"
#include "ComputePIStepSize.h"

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{

template <typename ContainerT, typename Field = double>
class StepInputs
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
      y_n_{},
      x_n_{static_cast<Field>(0)}
    {}

    void step(const Field h, const Field x_n, const std::size_t max_iterations = 100)
    {
      const std::size_t iterations {0};


    }

    ContainerT y_n_;
    ContainerT dydx_n_;
    
};

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_STEP_INPUTS_H
