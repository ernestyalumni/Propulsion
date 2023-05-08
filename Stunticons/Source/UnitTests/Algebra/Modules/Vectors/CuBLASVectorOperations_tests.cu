#include "Algebra/Modules/Vectors/Array.h"
#include "Algebra/Modules/Vectors/CuBLASVectorOperations.h"
#include "gtest/gtest.h"

#include <vector>

using Algebra::Modules::Vectors::Array;
using Algebra::Modules::Vectors::CuBLASVectorOperations;
using std::vector;

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
TEST(CuBLASVectorOperationsTests, Constructible)
{
  CuBLASVectorOperations X {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CuBLASVectorOperationsTests, Destructible)
{
  {
    CuBLASVectorOperations X {};
  }

  SUCCEED();
}

// Example values from CUDA Library Samples for cuBLAS.
// ref. https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLAS/Level-1/axpy/cublas_axpy_example.cu
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CuBLASVectorOperationsTests, AddsAndScalarMultipliesArrays)
{
  const vector<float> h_a {1.0, 2.0, 3.0, 4.0};
  vector<float> h_b {5.0, 6.0, 7.0, 8.0};

  Array X {h_a.size()};
  Array Y {h_b.size()};

  X.copy_host_input_to_device(h_a);
  Y.copy_host_input_to_device(h_b);

  CuBLASVectorOperations operations {};

  operations.scalar_multiply_and_add_vector(2.1, X, Y);

  // Synchronize to ensure the operation is completed.
  cudaDeviceSynchronize();

  vector<float> result (5, 0.0f);

  Y.copy_device_output_to_host(result);

  EXPECT_EQ(result.size(), 5);
  EXPECT_FLOAT_EQ(result.at(0), 7.10);
  EXPECT_FLOAT_EQ(result.at(1), 10.20);
  EXPECT_FLOAT_EQ(result.at(2), 13.30);
  EXPECT_FLOAT_EQ(result.at(3), 16.40);
  EXPECT_FLOAT_EQ(result.at(4), 0.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CuBLASVectorOperationsTests, TakesDotProduct)
{
  const vector<float> h_a {1.0, 2.0, 3.0, 4.0};
  const vector<float> h_b {5.0, 6.0, 7.0, 8.0};

  Array X {h_a.size()};
  Array Y {h_b.size()};

  X.copy_host_input_to_device(h_a);
  Y.copy_host_input_to_device(h_b);

  CuBLASVectorOperations operations {};

  // This should be a blocking (synchronous) call.
  const auto result = operations.dot_product(X, Y);

  EXPECT_TRUE(result.has_value());
  EXPECT_FLOAT_EQ(*result, 70.0);
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra
} // namespace GoogleUnitTests