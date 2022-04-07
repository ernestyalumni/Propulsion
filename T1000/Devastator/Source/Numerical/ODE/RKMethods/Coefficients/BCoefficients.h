#ifndef NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_B_COEFFICIENTS_H
#define NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_B_COEFFICIENTS_H

#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{
namespace Coefficients
{

template <std::size_t S, typename Field = double>
class BCoefficients : public std::array<Field, S>
{
  public:

    using Array = std::array<Field, S>;
    using Array::Array;

    BCoefficients() = default;

    BCoefficients(const std::initializer_list<Field>& b_coefficients)
    {
      std::copy(
        b_coefficients.begin(),
        b_coefficients.end(),
        this->begin());
    }

    // Copy constructor
    BCoefficients(const BCoefficients&) = default;

    //--------------------------------------------------------------------------
    /// \brief Copy assignment.
    /// cf. https://en.cppreference.com/w/cpp/language/copy_assignment
    //--------------------------------------------------------------------------
    BCoefficients& operator=(const BCoefficients&) = default;

    virtual ~BCoefficients() = default;

    Field get_ith_element(const std::size_t i) const
    {
      assert(i >= 1 && i <= S);

      return this->operator[](i - 1);
    }
};

//------------------------------------------------------------------------------
/// \details For Embedded Runge-Kutta methods, calculating the difference
/// between the higher-order formula and embedded formula.
//------------------------------------------------------------------------------
template <std::size_t S, typename Field = double>
using DeltaCoefficients = BCoefficients<S, Field>;

} // namespace Coefficients  
} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_B_COEFFICIENTS_H
