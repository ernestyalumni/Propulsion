#ifndef ALGEBRA_RINGS_COMPLEX_NUMBER_H
#define ALGEBRA_RINGS_COMPLEX_NUMBER_H

#include <type_traits>

namespace Algebra
{
namespace Rings
{

template <
  typename T = float,
  typename = std::enable_if_t<std::is_floating_point<T>::value>>
struct ComplexNumber
{
  __device__ ComplexNumber(const T a, const T b):
    x_{a},
    y_{b}
  {}

  inline __device__ float magnitude_squared(void)
  {
    return x_ * x_ + y_ * y_;
  }

  // Real number part.
  T x_;
  // Imaginary number part.
  T y_;
};

template <
  typename T = float,
  typename = std::enable_if_t<std::is_floating_point<T>::value>>
__device__ inline ComplexNumber operator+(
  const ComplexNumber& a,
  const ComplexNumber& b)
{
  return ComplexNumber{a.x_ + b.x_, a.y_ + b.y_};
}

template <
  typename T = float,
  typename = std::enable_if_t<std::is_floating_point<T>::value>>
__device__ inline ComplexNumber operator*(
  const ComplexNumber& a,
  const ComplexNumber& b)
{
  return ComplexNumber{
    a.x_ * b.x_ - a.y_ * b.y_,
    a.y_ * b.x_ + a.x_ * b.y_};
}

} // namespace Rings
} // namespace Algebra

#endif // ALGEBRA_RINGS_COMPLEX_NUMBER_H