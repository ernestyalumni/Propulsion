#ifndef ALGEBRA_MODULES_VECTORS_ARRAY_H
#define ALGEBRA_MODULES_VECTORS_ARRAY_H

#include "HostArrays.h"

#include <cstddef>
#include <vector>

namespace Algebra
{
namespace Modules
{
namespace Vectors
{

struct Array
{
  float* values_;
  const std::size_t number_of_elements_;

  Array(const std::size_t input_size = 50000);

  ~Array();

  bool copy_host_input_to_device(const HostArray& h_a);

  bool copy_device_output_to_host(HostArray& h_a);

  bool copy_host_input_to_device(const std::vector<float>& h_a);

  bool copy_device_output_to_host(std::vector<float>& h_a);

  //----------------------------------------------------------------------------
  /// \details cudaMemcpyAsync is non-blocking, so you need to handle
  /// synchronization explicitly. Also, it's stream-based execution, so you may
  /// need to specify a specific CUDA stream.
  /// Passing cudaMemcpyDefault for direction of copy is recommended, in which
  /// case type of transfer is inferred from pointer values. However,
  /// cudaMemcpyDefault only allowed on systems that support unified virtual
  /// addressing.
  /// cudaMemcpyAsync() can optionally be associated to a stream by passing a
  /// non-zero stream argument.
  /// \ref https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g85073372f776b4c4d5f89f7124b7bf79
  //----------------------------------------------------------------------------

  bool asynchronous_copy_host_input_to_device(
    const HostArray& h_a,
    cudaStream_t stream = 0);

  bool asynchronous_copy_device_output_to_host(
    HostArray& h_a,
    cudaStream_t stream = 0);

  std::size_t get_number_of_elements() const
  {
    return number_of_elements_;
  }
};

} // namespace Vectors
} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_VECTORS_ARRAY_H