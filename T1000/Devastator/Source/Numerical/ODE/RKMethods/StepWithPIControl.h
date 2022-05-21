#ifndef NUMERICAL_ODE_RK_METHODS_STEP_WITH_PI_CONTROL_H
#define NUMERICAL_ODE_RK_METHODS_STEP_WITH_PI_CONTROL_H

#include "CalculateNewYAndError.h"
#include "CalculateScaledError.h"
#include "ComputePIStepSize.h"

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
      y_n_{},
      x_n_{static_cast<Field>(0)}
    {}

    void step(const Field h, const Field x_n)

  private:

    CalculateNewYAndError<S, DerivativeType, Field>& new_y_and_err_;
    CalculateScaledError<Field> scaled_error_;
    ComputePIStepSize& pi_step_;
    ContainerT y_n_;
    Field x_n_;
};

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_STEP_WITH_PI_CONTROL_H
