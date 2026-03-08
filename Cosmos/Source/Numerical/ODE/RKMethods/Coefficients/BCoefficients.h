#ifndef NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_B_COEFFICIENTS_H
#define NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_B_COEFFICIENTS_H

#include <algorithm>
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

//------------------------------------------------------------------------------
/// \details Choice of std::vector<Field> over std::array is because it's an
/// aggregate and an aggregate shouldn't allocate memory dynamically.
/// \ref https://stackoverflow.com/questions/39548254/does-stdarray-guarantee-allocation-on-the-stack-only
//------------------------------------------------------------------------------
template <std::size_t S, typename Field = double>
class BCoefficients : public std::vector<Field>
{
  public:

    BCoefficients(const std::initializer_list<Field>& b_coefficients):
      std::vector<Field>(S)
    {
      assert(S == b_coefficients.size());

      //------------------------------------------------------------------------
      /// If you don't enforce size check before copying the input values,
      /// you'll get a Segmentation Fault during make when Google Unit Test's
      /// main() runs.
      //------------------------------------------------------------------------

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

    Field& ith_element(const std::size_t i)
    {
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
