#ifndef NUMERICAL_ODE_RK_METHODS_HIGHER_ORDER_STEP_WITH_PI_CONTROL_H
#define NUMERICAL_ODE_RK_METHODS_HIGHER_ORDER_STEP_WITH_PI_CONTROL_H

#include "CalculateError.h"
#include "CalculateNewY.h"
#include "ComputePIStepSize.h"
#include "PIStepSizeControl.h"
#include "StepInputs.h"

#include <cstdint>
#include <stdexcept>

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{

//------------------------------------------------------------------------------
/// \details It seems that this class is distinctly different from
/// StepWithPIControl because of
/// - the calculation of 2 different error values,
/// - the use of the so-called bhh coefficients in that calculation,
/// - the tolerances, a_tolerance, r_tolerance, depending on the component
/// - index, as opposed to being the same value for each component of the y
/// vector.
//------------------------------------------------------------------------------
template <
  std::size_t S,
  std::size_t BHHSize,
  typename DerivativeType,
  typename ContainerT,
  typename Field = double>
class HigherOrderStepWithPIControl
{
  public:

    HigherOrderStepWithPIControl(
      CalculateNewY<S, DerivativeType, Field>& new_y,
      CalculateError<S, BHHSize, ContainerT, Field>& error,
      const ComputePIStepSize<Field>& pi_step
      ):
      new_y_{new_y},
      error_{error},
      pi_step_{pi_step},
      pi_control_{}
    {}

    HigherOrderStepWithPIControl(
      CalculateNewY<S, DerivativeType, Field>&& new_y,
      CalculateError<S, BHHSize, ContainerT, Field>&& error,
      const ComputePIStepSize<Field>&& pi_step
      ):
      new_y_{new_y},
      error_{error},
      pi_step_{pi_step},
      pi_control_{}
    {}

    //--------------------------------------------------------------------------
    /// \return h, the step value used to compute the new y with, *not* the h
    /// value computed for the next step.
    //--------------------------------------------------------------------------
    template <std::size_t N>
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

        y_out = new_y_.calculate_new_y(
          h,
          inputs.x_n_,
          inputs.y_n_,
          inputs.dydx_n_,
          inputs.k_coefficients_);

        //----------------------------------------------------------------------
        /// You *must* use this syntax for calling templated class member
        /// functions of class members; otherwise you obtain this error in
        /// compilation: error: invalid operands of types ‘<unresolved
        /// overloaded function type>’
        //----------------------------------------------------------------------
        error = error_.template calculate_scaled_error<N>(
          inputs.y_n_,
          y_out,
          inputs.k_coefficients_,
          h);

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

    ContainerT calculate_derivative(
      const Field x,
      const ContainerT& y)
    {
      return new_y_.calculate_derivative(x, y);
    }

  private:

    CalculateNewY<S, DerivativeType, Field>& new_y_;
    CalculateError<S, BHHSize, ContainerT, Field> error_;
    ComputePIStepSize<Field> pi_step_;
    PIStepSizeControl<Field> pi_control_;
};

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_HIGHER_ORDER_STEP_WITH_PI_CONTROL_H
