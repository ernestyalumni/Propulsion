#ifndef NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_C_COEFFICIENTS_H
#define NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_C_COEFFICIENTS_H

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
class CCoefficients : public std::array<Field, S - 1>
{
  public:

    using Array = std::array<Field, S - 1>;
    using Array::Array;

    CCoefficients(const std::initializer_list<Field>& c_coefficients)
    {
      std::copy(
        c_coefficients.begin(),
        c_coefficients.end(),
        this->begin());
    }

    // Copy constructor
    CCoefficients(const CCoefficients&) = default;

    //--------------------------------------------------------------------------
    /// \brief Copy assignment.
    /// cf. https://en.cppreference.com/w/cpp/language/copy_assignment
    //--------------------------------------------------------------------------
    CCoefficients& operator=(const CCoefficients&) = default;

    virtual ~CCoefficients() = default;

    Field get_ith_element(const std::size_t i) const
    {
      assert(i >= 2 && i <= S);

      return this->operator[](i - 2);
    }
};

} // namespace Coefficients  
} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_C_COEFFICIENTS_H
