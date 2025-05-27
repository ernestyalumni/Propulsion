#ifndef TESTING_MANIFOLDS_OPERATORS_FINITE_DIFFERENCE_2D_DIRECTIONAL_DERIVATIVES_KERNEL_H
#define TESTING_MANIFOLDS_OPERATORS_FINITE_DIFFERENCE_2D_DIRECTIONAL_DERIVATIVES_KERNEL_H

#include "Manifolds/Operators/FiniteDifference/C_Nu_Coefficients.h"
#include "Manifolds/Operators/FiniteDifference/2D/DirectionalDerivatives.h"
#include "Utilities/HandleUnsuccessfulCUDACall.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace Testing
{
namespace Manifolds
{
namespace Operators
{
namespace FiniteDifference
{
namespace TwoDimensional
{

template <typename FPT, typename CompoundFPT, int NU>
__global__ void directional_derivatives_kernel(
  FPT* result,
  FPT (*stencil)[2])
{
  using ::Manifolds::Operators::FiniteDifference::cnu_coefficients_first_order;

  FPT c_nus_x[4] {
    cnu_coefficients_first_order<CompoundFPT>[0].x,
    cnu_coefficients_first_order<CompoundFPT>[1].x,
    cnu_coefficients_first_order<CompoundFPT>[2].x,
    cnu_coefficients_first_order<CompoundFPT>[3].x
  };

  *result = ::Manifolds::Operators::FiniteDifference::TwoDimensional::directional_derivative<FPT, NU>(
    stencil,
    c_nus_x);
}

template <typename FPT, typename CompoundFPT, int NU>
FPT test_directional_derivatives(
  FPT (*stencil)[2])
{
  FPT* result {};
  FPT result_host {};
  Utilities::HandleUnsuccessfulCUDACall handle_malloc {
    "Failed to allocate memory for result"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_malloc,
    cudaMalloc(&result, sizeof(FPT)));

  if (!handle_malloc.is_cuda_success())
  {
    throw std::runtime_error(std::string(handle_malloc.get_error_message()));
  }

  directional_derivatives_kernel<FPT, CompoundFPT, NU><<<1, 1>>>(
    result,
    stencil);

  Utilities::HandleUnsuccessfulCUDACall handle_memcpy {
    "Failed to copy result from device to host"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_memcpy,
    cudaMemcpy(&result_host, result, sizeof(FPT), cudaMemcpyDeviceToHost));

  cudaFree(result);

  if (!handle_memcpy.is_cuda_success())
  {
    throw std::runtime_error(std::string(handle_memcpy.get_error_message()));
  }

  return result_host;
}

} // namespace TwoDimensional
} // namespace FiniteDifference
} // namespace Operators
} // namespace Manifolds
} // namespace Testing

#endif // TESTING_MANIFOLDS_OPERATORS_FINITE_DIFFERENCE_2D_DIRECTIONAL_DERIVATIVES_KERNEL_H
