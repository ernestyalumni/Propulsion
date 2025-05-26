#include "DirectionalDerivativesKernel.h"

#include <cuda_fp16.h> // __half

namespace Utilities
{
namespace Testing
{
namespace Manifolds
{
namespace Operators
{
namespace FiniteDifference
{

template __global__ void directional_derivatives_kernel<float, float2, 1>(
  float* result,
  float (*stencil)[2]);

template __global__ void directional_derivatives_kernel<float, float2, 2>(
  float* result,
  float (*stencil)[2]);

template __global__ void directional_derivatives_kernel<float, float2, 3>(
  float* result,
  float (*stencil)[2]);

template __global__ void directional_derivatives_kernel<float, float2, 4>(
  float* result,
  float (*stencil)[2]);

template __global__ void directional_derivatives_kernel<double, double2, 1>(
  double* result,
  double (*stencil)[2]);

template __global__ void directional_derivatives_kernel<double, double2, 2>(
  double* result,
  double (*stencil)[2]);

template __global__ void directional_derivatives_kernel<double, double2, 3>(
  double* result,
  double (*stencil)[2]);

template __global__ void directional_derivatives_kernel<double, double2, 4>(
  double* result,
  double (*stencil)[2]);

template __global__ void directional_derivatives_kernel<__half, __half2, 1>(
  __half* result,
  __half (*stencil)[2]);

template __global__ void directional_derivatives_kernel<__half, __half2, 2>(
  __half* result,
  __half (*stencil)[2]);

template __global__ void directional_derivatives_kernel<__half, __half2, 3>(
  __half* result,
  __half (*stencil)[2]);

template __global__ void directional_derivatives_kernel<__half, __half2, 4>(
  __half* result,
  __half (*stencil)[2]);

template <>
float test_directional_derivatives<float, float2, 1>(float (*stencil)[2]);

template <>
float test_directional_derivatives<float, float2, 2>(float (*stencil)[2]);

template <>
float test_directional_derivatives<float, float2, 3>(float (*stencil)[2]);

template <>
float test_directional_derivatives<float, float2, 4>(float (*stencil)[2]);

template <>
double test_directional_derivatives<double, double2, 1>(double (*stencil)[2]);

template <>
double test_directional_derivatives<double, double2, 2>(double (*stencil)[2]);

template <>
double test_directional_derivatives<double, double2, 3>(double (*stencil)[2]);

template <>
double test_directional_derivatives<double, double2, 4>(double (*stencil)[2]);

template <>
__half test_directional_derivatives<__half, __half2, 1>(__half (*stencil)[2]);

template <>
__half test_directional_derivatives<__half, __half2, 2>(__half (*stencil)[2]);

template <>
__half test_directional_derivatives<__half, __half2, 3>(__half (*stencil)[2]);

template <>
__half test_directional_derivatives<__half, __half2, 4>(__half (*stencil)[2]);

} // namespace FiniteDifference
} // namespace Operators
} // namespace Manifolds
} // namespace Testing
} // namespace Utilities

