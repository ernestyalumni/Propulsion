#ifndef NUMERICAL_ODE_RK_METHODS_STEP_INPUTS_H
#define NUMERICAL_ODE_RK_METHODS_STEP_INPUTS_H

#include "Coefficients/KCoefficients.h"

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{

template <std::size_t S, typename ContainerT, typename Field = double>
class StepInputs
{
  public:

    StepInputs():
      k_coefficients_{},
      y_n_{},
      dydx_n_{},
      h_n_{},
      x_n_{}
    {}

    StepInputs(
      const ContainerT& y_n,
      const ContainerT& dydx_n,
      const Field h_n,
      const Field x_n
      ):
      k_coefficients_{},
      y_n_{y_n},
      dydx_n_{dydx_n},
      h_n_{h_n},
      x_n_{x_n}
    {}

    virtual ~StepInputs() = default;

    Coefficients::KCoefficients<S, ContainerT> k_coefficients_;
    ContainerT y_n_;
    ContainerT dydx_n_;
    Field h_n_;
    Field x_n_;
};

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_STEP_INPUTS_H
