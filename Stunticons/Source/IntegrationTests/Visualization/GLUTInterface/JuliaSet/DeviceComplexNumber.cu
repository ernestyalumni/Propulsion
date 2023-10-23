#include "DeviceComplexNumber.h"

#include <cuda_runtime.h>

namespace IntegrationTests
{
namespace Visualization
{
namespace GLUTInterface
{

/*
__device__ DeviceComplexNumber& DeviceComplexNumber::operator+(
  const DeviceComplexNumber& a)
{
  r_ += a.r_;
  i_ += a.i_;

  return *this;
}

__device__ DeviceComplexNumber& DeviceComplexNumber::operator*(
  const DeviceComplexNumber& a)
{
  r_ = r_ * a.r_ - i_ * a.i_;
  i_ = i_ * a.r_ + r_ * a.i_;

  return *this;
}
*/

} // namespace GLUTInterface
} // namespace Visualization
} // namespace IntegrationTests