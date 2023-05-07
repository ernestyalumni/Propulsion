#include "Algebra/Modules/Vectors/DeviceVectorAdditionArrays.h"
#include "Algebra/Modules/Vectors/HostVectorAdditionArrays.h"
#include "Algebra/Modules/Vectors/VectorAddition.h"
#include "gtest/gtest.h"

#include <cstddef>
#include <cuda_runtime.h>
#include <type_traits>

using Algebra::Modules::Vectors::DeviceVectorAdditionArrays;
using Algebra::Modules::Vectors::HostVectorAdditionArrays;
using Algebra::Modules::Vectors::copy_device_output_to_host;
using Algebra::Modules::Vectors::copy_host_input_to_device;
using Algebra::Modules::Vectors::vector_addition;
using std::size_t;

namespace GoogleUnitTests
{
namespace Algebra
{
namespace Modules
{
namespace Vectors
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(VectorAdditionTests, VectorAdditionAddsAsKernel)
{
  HostVectorAdditionArrays h_arrays {};

  for (std::size_t i {0}; i < h_arrays.number_of_elements_; ++i)
  {
    h_arrays.h_A_[i] = static_cast<float>(i);
    h_arrays.h_B_[i] = static_cast<float>(i) + 2.0f;
  }

  DeviceVectorAdditionArrays d_arrays {};

  copy_host_input_to_device(h_arrays, d_arrays);

  constexpr size_t threads_per_block {256};
  const size_t blocks_per_grid {
    (h_arrays.number_of_elements_ + threads_per_block - 1) / threads_per_block};

  vector_addition<<<blocks_per_grid, threads_per_block>>>(
    d_arrays.d_A_,
    d_arrays.d_B_,
    d_arrays.d_C_,
    d_arrays.number_of_elements_);

  const cudaError_t err {cudaGetLastError()};

  if (err != cudaSuccess)
  {
    FAIL();
  }

  copy_device_output_to_host(d_arrays, h_arrays);

  for (std::size_t i {0}; i < h_arrays.number_of_elements_; ++i)
  {
    EXPECT_FLOAT_EQ(h_arrays.h_C_[i], 2.0 * static_cast<float>(i) + 2.0f);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(VectorAdditionTests, VectorAdditionAdds)
{
  HostVectorAdditionArrays h_arrays {};

  for (std::size_t i {0}; i < h_arrays.number_of_elements_; ++i)
  {
    h_arrays.h_A_[i] = static_cast<float>(i);
    h_arrays.h_B_[i] = 2.0 * static_cast<float>(i) + 3.0f;
  }

  DeviceVectorAdditionArrays d_arrays {};

  vector_addition(h_arrays, d_arrays);

  for (std::size_t i {0}; i < h_arrays.number_of_elements_; ++i)
  {
    EXPECT_FLOAT_EQ(h_arrays.h_C_[i], 3.0 * static_cast<float>(i) + 3.0f);
  }
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra
} // namespace GoogleUnitTests