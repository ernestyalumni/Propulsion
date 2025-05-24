#include "C_Nu_Coefficients.h"

namespace Manifolds
{
namespace Operators
{
namespace FiniteDifference
{

// Explicit template instantiation.

// Define the static constant arrays
template<>
__constant__ float2 cnu_coefficients_first_order<float2>[4];

template<>
__constant__ float2 cnu_coefficients_second_order<float2>[4];

// Explicit instantiation of the template function
template void auxiliary_set_first_order_coefficients<float, float2, 4>(
  const float[2],
  const float[4],
  float2*);

} // namespace FiniteDifference
} // namespace Operators
} // namespace Manifolds
