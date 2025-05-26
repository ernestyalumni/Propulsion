#include "DirectionalDerivatives.h"

#include <cuda_fp16.h> // __half

namespace Manifolds
{
namespace Operators
{
namespace FiniteDifference
{

template __device__ float directional_derivative<float, 1>(
    float stencil[1][2],
    float c_nus[4]);

template __device__ float directional_derivative<float, 2>(
    float stencil[2][2],
    float c_nus[4]);

template __device__ float directional_derivative<float, 3>(
    float stencil[3][2],
    float c_nus[4]);

template __device__ float directional_derivative<float, 4>(
    float stencil[4][2],
    float c_nus[4]);

template __device__ double directional_derivative<double, 1>(
    double stencil[1][2],
    double c_nus[4]);

template __device__ double directional_derivative<double, 2>(
    double stencil[2][2],
    double c_nus[4]);

template __device__ double directional_derivative<double, 3>(
    double stencil[3][2],
    double c_nus[4]);

template __device__ double directional_derivative<double, 4>(
    double stencil[4][2],
    double c_nus[4]);

template __device__ __half directional_derivative<__half, 1>(
    __half stencil[1][2],
    __half c_nus[4]);

template __device__ __half directional_derivative<__half, 2>(
    __half stencil[2][2],
    __half c_nus[4]);

template __device__ __half directional_derivative<__half, 3>(
    __half stencil[3][2],
    __half c_nus[4]);

template __device__ __half directional_derivative<__half, 4>(
    __half stencil[4][2],
    __half c_nus[4]);

} // namespace FiniteDifference
} // namespace Operators
} // namespace Manifolds