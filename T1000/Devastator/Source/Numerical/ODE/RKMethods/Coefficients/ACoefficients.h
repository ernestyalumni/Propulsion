#ifndef NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_A_COEFFICIENTS_H
#define NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_A_COEFFICIENTS_H

#include <array>
#include <cstddef>

namespace Numerical
{
namespace ODE
{
namespace RKMethods
{
namespace Coefficients
{

template <std::size_t M, typename Field = double>
class ACoefficients : public std::array<Field, M * (M - 1)>
{
  public:

    using Array = std::array<Field, M * (M - 1)>;
    using Array::Array;

    Field get_ij_element(const std::size_t i, const std::size_t j) const
    {
      assert(i > j && j >= 1 && i <= M)
      {
        std::size_t n {i - 2};
        n = n * (n + 1) / 2;

        return this->operator[n + (j - 1)];
      }
    }
};

} // namespace Coefficients  
} // namespace RKMethods
} // namespace ODE
} // namespace Numerical

#endif // NUMERICAL_ODE_RK_METHODS_COEFFICIENTS_A_COEFFICIENTS_H
