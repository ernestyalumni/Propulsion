#include "C_Nu_Coefficients.h"

#include <cuda_fp16.h>
// TODO: Decide if we want to use bfloat16.
// #include <cuda_bf16.h>

namespace Manifolds
{
namespace Operators
{
namespace FiniteDifference
{

// TODO: Decide if we want to use bfloat16.
/*
template<>
__constant__ __nv_bfloat162 cnu_coefficients_first_order<__nv_bfloat162>[4];

template<>
__constant__ __nv_bfloat162 cnu_coefficients_second_order<__nv_bfloat162>[4];
*/

// Explicit instantiation of the template function

template void auxiliary_set_first_order_coefficients<float, float2, 4>(
  const float[2],
  const float[4],
  float2*);

template void auxiliary_set_second_order_coefficients<float, float2, 4>(
  const float[2],
  const float[4],
  float2*);

template void auxiliary_set_first_order_coefficients<double, double2, 4>(
  const double[2],
  const double[4],
  double2*);

template void auxiliary_set_second_order_coefficients<double, double2, 4>(
  const double[2],
  const double[4],
  double2*);

template void auxiliary_set_first_order_coefficients<__half, __half2, 4>(
  const __half[2],
  const __half[4],
  __half2*);

template void auxiliary_set_second_order_coefficients<__half, __half2, 4>(
  const __half[2],
  const __half[4],
  __half2*);

// TODO: Decide if we want to use bfloat16.
/*
template void auxiliary_set_first_order_coefficients<
  __nv_bfloat162,
  __nv_bfloat162,
  4>(
    const __nv_bfloat162[2],
    const __nv_bfloat162[4],
  __nv_bfloat162*);

template void auxiliary_set_second_order_coefficients<
  __nv_bfloat162,
  __nv_bfloat162,
  4>(
    const __nv_bfloat162[2],
    const __nv_bfloat162[4],
    __nv_bfloat162*);
*/

} // namespace FiniteDifference
} // namespace Operators
} // namespace Manifolds
