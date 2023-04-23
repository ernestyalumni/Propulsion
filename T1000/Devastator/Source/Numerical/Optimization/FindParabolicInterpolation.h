#ifndef NUMERICAL_OPTIMIZATION_FIND_PARABOLIC_INTERPOLATION_H
#define NUMERICAL_OPTIMIZATION_FIND_PARABOLIC_INTERPOLATION_H

#include <math.h> // std::copysign
#include <type_traits>

namespace Numerical
{
namespace Optimization
{

struct FindParabolicInterpolation
{
  template <
    typename T = double,
    // For the case when it's false that it's a floating point,
    // std::enable_if_t<false> is ill-formed because std::enable_if<false>
    // specialization does not define a nested type called ::type.
    typename = typename std::enable_if_t<std::is_floating_point<T>::value>
    >
  struct Results
  {
    T extrema_position_;
    T second_derivative_;
  };

  template <
    typename T = double,
    // For the case when it's false that it's a floating point,
    // std::enable_if_t<false> is ill-formed because std::enable_if<false>
    // specialization does not define a nested type called ::type.
    typename = typename std::enable_if_t<std::is_floating_point<T>::value>
    >
  static Results find_extrema(
    const T a,
    const T b,
    const T c,
    const T fa,
    const T fb,
    const T fc)
  {
    static constexpr T tiny {1.0e-20};

    const T q {(b-a)*(fc - fa)};
    const T r {(c-a)*(fb - fa)};

    const T u {
      0.5 * ((a + b) - r * (c - b) / (
        std::copysign(std::max(std::abs(q - r), tiny), q - r)))};

    const T denominator {(b-a)*(c-a)*(c-b)};

    const T double_derivative {
      (q - r) / (
        std::copysign(std::max(std::abs(denominator), tiny), denominator))};
  
    return Results{u, double_derivative};
  }
};

} // namespace Optimization
} // namespace Numerical

#endif // NUMERICAL_OPTIMIZATION_FIND_PARABOLIC_INTERPOLATION_H

