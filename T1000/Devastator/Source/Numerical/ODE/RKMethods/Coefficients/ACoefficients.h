#ifndef NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_A_COEFFICIENTS_H
#define NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_A_COEFFICIENTS_H

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
class ACoefficients : public std::array<Field, S * (S - 1) / 2>
{
  public:

    using Array = std::array<Field, S * (S - 1) / 2>;
    using Array::Array;

    ACoefficients(const std::initializer_list<Field>& a_coefficients)
    {
      std::copy(
        a_coefficients.begin(),
        a_coefficients.end(),
        this->begin());
    }

    // Copy constructor
    ACoefficients(const ACoefficients&) = default;

    //--------------------------------------------------------------------------
    /// \brief Copy assignment, when copy-and-swap idiom isn't used.
    /// TODO: Add more to data member such as a unique_ptr to int array, for
    /// demonstration.
    /// cf. https://en.cppreference.com/w/cpp/language/copy_assignment
    //--------------------------------------------------------------------------
    ACoefficients& operator=(const ACoefficients&) = default;

    virtual ~ACoefficients() = default;

    Field get_ij_element(const std::size_t i, const std::size_t j) const
    {
      assert(i > j && j >= 1 && i <= S);

      std::size_t n {i - 2};
      n = n * (n + 1) / 2;

      return this->operator[](n + (j - 1));
    }
};

} // namespace Coefficients  
} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_A_COEFFICIENTS_H
