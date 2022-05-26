#ifndef NUMERICAL_ODE_RK_METHODS_CALCULATE_DENSE_OUTPUT_COEFFICIENT_H
#define NUMERICAL_ODE_RK_METHODS_CALCULATE_DENSE_OUTPUT_COEFFICIENT_H

#include "Algebra/Modules/Vectors/NVector.h"
#include "Coefficients/BCoefficients.h"
#include "Coefficients/KCoefficients.h"

#include <cstddef>
#include <type_traits>
#include <valarray>

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{

//------------------------------------------------------------------------------
/// \ref pp. 919, 17.2 Adaptive Stepsize Control for Runge-Kutta, Numerical
/// Recipes, last line of Doub StepperDopr5<D>::dense_out(const Int i,const Doub
/// x,const Doub h) function for stepperdopr5.h.
//------------------------------------------------------------------------------

template <std::size_t S, typename Field = double>
class CalculateDenseOutputCoefficient
{
  public:

    CalculateDenseOutputCoefficient() = delete;

    CalculateDenseOutputCoefficient(
      const Coefficients::DeltaCoefficients<S, Field>& dense_coefficients
      ):
      dense_coefficients_{dense_coefficients}
    {}

    virtual ~CalculateDenseOutputCoefficient() = default;

    template <typename ContainerT>
    ContainerT operator()(
      const Coefficients::KCoefficients<S, ContainerT>& k_coefficients,
      const Field theta,
      const Field h)
    {
      ContainerT dense_coefficient {
        dense_coefficients_.get_ith_element(1) *
          k_coefficients.get_ith_coefficient(1)};

      for (std::size_t j {2}; j <= S; ++j)
      {
        dense_coefficient += dense_coefficients_.get_ith_element(j) *
          k_coefficients.get_ith_coefficient(j);
      }

      return theta * theta * (1.0 - theta) * (1.0 - theta) * h * 
        dense_coefficient;
    }
};

} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_CALCULATE_DENSE_OUTPUT_COEFFICIENT_H
