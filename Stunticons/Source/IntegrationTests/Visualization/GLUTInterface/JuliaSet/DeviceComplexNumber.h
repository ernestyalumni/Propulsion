#ifndef INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_JULIA_SET_DEVICE_COMPLEX_NUMBER_H
#define INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_JULIA_SET_DEVICE_COMPLEX_NUMBER_H

#include <cuda_runtime.h>

namespace IntegrationTests
{
namespace Visualization
{
namespace GLUTInterface
{

struct DeviceComplexNumber
{
  __device__ DeviceComplexNumber(const float a, const float b):
    r_{a},
    i_{b}
  {}

  inline __device__ float magnitude_squared(void)
  {
    return r_ * r_ + i_ * i_;
  }

  //__device__ DeviceComplexNumber& operator+(const DeviceComplexNumber& a);

  //__device__ DeviceComplexNumber& operator*(const DeviceComplexNumber& b);

  // Real number part.
  float r_;
  // Imaginary number part.
  float i_;
};

inline __device__ DeviceComplexNumber operator+(
  const DeviceComplexNumber& a,
  const DeviceComplexNumber& b)
{
  return DeviceComplexNumber{a.r_ + b.r_, a.i_ + b.i_};
}

inline __device__ DeviceComplexNumber operator*(
  const DeviceComplexNumber& a,
  const DeviceComplexNumber& b)
{
  return DeviceComplexNumber{
    a.r_ * b.r_ - a.i_ * b.i_,
    a.i_ * b.r_ + a.r_ * b.i_};
}

} // namespace GLUTInterface
} // namespace Visualization
} // namespace IntegrationTests

#endif // INTEGRATION_TESTS_VISUALIZATION_GLUT_INTERFACE_JULIA_SET_DEVICE_COMPLEX_NUMBER_H