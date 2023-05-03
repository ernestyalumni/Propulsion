#include "DeviceVectorAdditionArrays.h"

#include <cstddef> // std::size_t
#include <cstdlib> // free
#include <iostream> // std::cerr
#include <stdexcept>

using std::cerr;
using std::size_t;

namespace Algebra
{
namespace Modules
{
namespace Vectors
{

DeviceVectorAdditionArrays::DeviceVectorAdditionArrays(
  const std::size_t input_size
  ):
  number_of_elements_{input_size},
  d_A_{nullptr},
  d_B_{nullptr},
  d_C_{nullptr}
{
  const size_t size_in_bytes {input_size * sizeof(float)};

  const cudaError_t err_A {
    cudaMalloc(reinterpret_cast<void**>(&d_A_), size_in_bytes)};
  const cudaError_t err_B {
    cudaMalloc(reinterpret_cast<void**>(&d_B_), size_in_bytes)};
  const cudaError_t err_C {
    cudaMalloc(reinterpret_cast<void**>(&d_C_), size_in_bytes)};

  if (err_A != cudaSuccess)
  {
    cerr << "Failed to allocate device array A (error code " <<
      cudaGetErrorString(err_A) << ")!\n";
  }
  if (err_B != cudaSuccess)
  {
    cerr << "Failed to allocate device array B (error code " <<
      cudaGetErrorString(err_B) << ")!\n";
  }
  if (err_C != cudaSuccess)
  {
    cerr << "Failed to allocate device array C (error code " <<
      cudaGetErrorString(err_C) << ")!\n";
  }

  if (err_A != cudaSuccess || err_B != cudaSuccess || err_C != cudaSuccess)
  {
    throw std::runtime_error("Failed to allocate device array");
  }
}

DeviceVectorAdditionArrays::~DeviceVectorAdditionArrays()
{
  const cudaError_t err_A {cudaFree(d_A_)};
  const cudaError_t err_B {cudaFree(d_B_)};
  const cudaError_t err_C {cudaFree(d_C_)};

  if (err_A != cudaSuccess)
  {
    cerr << "Failed to free device array A (error code " <<
      cudaGetErrorString(err_A) << ")!\n";
  }
  if (err_B != cudaSuccess)
  {
    cerr << "Failed to free device array B (error code " <<
      cudaGetErrorString(err_B) << ")!\n";
  }
  if (err_C != cudaSuccess)
  {
    cerr << "Failed to free device array C (error code " <<
      cudaGetErrorString(err_C) << ")!\n";
  }

  // We choose not to throw upon a failed garbage clean up.
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra