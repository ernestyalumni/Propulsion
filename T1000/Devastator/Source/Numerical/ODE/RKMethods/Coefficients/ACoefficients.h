#ifndef NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_A_COEFFICIENTS_H
#define NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_A_COEFFICIENTS_H

#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <vector>

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{
namespace Coefficients
{

template <std::size_t S, typename Field = double>
class ACoefficients : public std::vector<Field>
{
  public:

    ACoefficients(const std::initializer_list<Field>& a_coefficients):
      std::vector<Field>(S * (S - 1) / 2)
    {
      std::copy(
        a_coefficients.begin(),
        a_coefficients.end(),
        this->begin());
    }

    // Copy constructor
    ACoefficients(const ACoefficients&) = default;

    //--------------------------------------------------------------------------
    /// \brief Copy assignment.
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
