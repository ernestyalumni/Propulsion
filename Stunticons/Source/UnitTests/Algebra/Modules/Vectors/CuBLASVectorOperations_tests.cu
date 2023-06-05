#include "Algebra/Modules/Matrices/CompressedSparseRow.h"
#include "Algebra/Modules/Vectors/Array.h"
#include "Algebra/Modules/Vectors/CuBLASVectorOperations.h"
#include "gtest/gtest.h"

#include <vector>

using Algebra::Modules::Matrices::SparseMatrices::DenseVector;
using Algebra::Modules::Matrices::SparseMatrices::DoubleDenseVector;
using Algebra::Modules::Vectors::Array;
using Algebra::Modules::Vectors::CuBLASVectorOperations;
using Algebra::Modules::Vectors::DoubleArray;
using Algebra::Modules::Vectors::DoubleCuBLASVectorOperations;
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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CuBLASVectorOperationsTests, ScalarMultiplies)
{
  const vector<float> h_a {1.0, 2.0, 3.0, 4.0};
  Array X {h_a.size()};

  X.copy_host_input_to_device(h_a);

  CuBLASVectorOperations operations {};

  EXPECT_TRUE(operations.scalar_multiply(2.2, X));

  vector<float> result (4, 0.0f);

  X.copy_device_output_to_host(result);

  EXPECT_FLOAT_EQ(result.at(0), 2.2);
  EXPECT_FLOAT_EQ(result.at(1), 4.4);
  EXPECT_FLOAT_EQ(result.at(2), 6.6);
  EXPECT_FLOAT_EQ(result.at(3), 8.8);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CuBLASVectorOperationsTests, ScalarMultipliesWithDenseVector)
{
  const vector<float> h_a {1.0, 2.0, 3.0, 4.0};
  DenseVector X {h_a.size()};

  X.copy_host_input_to_device(h_a);

  CuBLASVectorOperations operations {};

  EXPECT_TRUE(operations.scalar_multiply(2.2, X));

  vector<float> result (4, 0.0f);

  X.copy_device_output_to_host(result);

  EXPECT_FLOAT_EQ(result.at(0), 2.2);
  EXPECT_FLOAT_EQ(result.at(1), 4.4);
  EXPECT_FLOAT_EQ(result.at(2), 6.6);
  EXPECT_FLOAT_EQ(result.at(3), 8.8);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CuBLASVectorOperationsTests, CopyCopiesDenseVectorToDenseVector)
{
  const vector<float> h_a {69.0, 42.0, 888.0, 420.0};
  DenseVector a {h_a.size()};

  const vector<float> h_b {1.0, 2.0, 3.0, 4.0};
  DenseVector b {h_b.size()};

  a.copy_host_input_to_device(h_a);
  b.copy_host_input_to_device(h_b);

  vector<float> result (4, 0.0f);
  b.copy_device_output_to_host(result);

  EXPECT_FLOAT_EQ(result.at(0), 1.0);
  EXPECT_FLOAT_EQ(result.at(1), 2.0);
  EXPECT_FLOAT_EQ(result.at(2), 3.0);
  EXPECT_FLOAT_EQ(result.at(3), 4.0);

  CuBLASVectorOperations operations {};

  EXPECT_TRUE(operations.copy(a, b));

  b.copy_device_output_to_host(result);

  EXPECT_FLOAT_EQ(result.at(0), 69.0);
  EXPECT_FLOAT_EQ(result.at(1), 42.0);
  EXPECT_FLOAT_EQ(result.at(2), 888.0);
  EXPECT_FLOAT_EQ(result.at(3), 420.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CuBLASVectorOperationsTests, CopyCopiesDenseVectorToArray)
{
  const vector<float> h_a {69.0, 42.0, 888.0, 420.0};
  DenseVector a {h_a.size()};

  const vector<float> h_b {1.0, 2.0, 3.0, 4.0};
  Array b {h_b.size()};

  a.copy_host_input_to_device(h_a);
  b.copy_host_input_to_device(h_b);

  vector<float> result (4, 0.0f);
  b.copy_device_output_to_host(result);

  EXPECT_FLOAT_EQ(result.at(0), 1.0);
  EXPECT_FLOAT_EQ(result.at(1), 2.0);
  EXPECT_FLOAT_EQ(result.at(2), 3.0);
  EXPECT_FLOAT_EQ(result.at(3), 4.0);

  CuBLASVectorOperations operations {};

  EXPECT_TRUE(operations.copy(a, b));

  b.copy_device_output_to_host(result);

  EXPECT_FLOAT_EQ(result.at(0), 69.0);
  EXPECT_FLOAT_EQ(result.at(1), 42.0);
  EXPECT_FLOAT_EQ(result.at(2), 888.0);
  EXPECT_FLOAT_EQ(result.at(3), 420.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CuBLASVectorOperationsTests, GetNormGetsNormForArray)
{
  const vector<float> h_a {10.0, 10.0, 3.0, 4.0};
  Array X {h_a.size()};

  X.copy_host_input_to_device(h_a);

  CuBLASVectorOperations operations {};

  const auto result = operations.get_norm(X);

  EXPECT_TRUE(result.has_value());

  EXPECT_FLOAT_EQ(*result, 15.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CuBLASVectorOperationsTests, GetNormGetsNormForDenseArray)
{
  const vector<float> h_a {69.0, 42.0, 888.0, 420.0};
  DenseVector X {h_a.size()};

  X.copy_host_input_to_device(h_a);

  CuBLASVectorOperations operations {};

  const auto result = operations.get_norm(X);

  EXPECT_TRUE(result.has_value());

  EXPECT_FLOAT_EQ(*result, 985.6312697961647);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DoubleCuBLASVectorOperationsTests, TakesDotProductsOfDoubleArrays)
{
  const vector<double> h_a {1.0, 2.0, 3.0, 4.0};
  const vector<double> h_b {5.0, 6.0, 7.0, 8.0};

  DoubleArray X {h_a.size()};
  DoubleArray Y {h_b.size()};

  X.copy_host_input_to_device(h_a);
  Y.copy_host_input_to_device(h_b);

  DoubleCuBLASVectorOperations operations {};

  // This should be a blocking (synchronous) call.
  const auto result = operations.dot_product(X, Y);

  EXPECT_TRUE(result.has_value());
  EXPECT_DOUBLE_EQ(*result, 70.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(
  DoubleCuBLASVectorOperationsTests,
  TakesDotProductOfDoubleDenseArrayAndDoubleArray)
{
  const vector<double> h_a {1.0, 2.0, 3.0, 4.0};
  const vector<double> h_b {5.0, 6.0, 7.0, 8.0};

  DoubleDenseVector X {h_a.size()};
  DoubleArray Y {h_b.size()};

  X.copy_host_input_to_device(h_a);
  Y.copy_host_input_to_device(h_b);

  DoubleCuBLASVectorOperations operations {};

  // This should be a blocking (synchronous) call.
  const auto result = operations.dot_product(X, Y);

  EXPECT_TRUE(result.has_value());
  EXPECT_DOUBLE_EQ(*result, 70.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DoubleCuBLASVectorOperationsTests, GetNormGetsNormForDoubleArray)
{
  const vector<double> h_a {10.0, 10.0, 3.0, 4.0};
  DoubleArray X {h_a.size()};

  X.copy_host_input_to_device(h_a);

  DoubleCuBLASVectorOperations operations {};

  const auto result = operations.get_norm(X);

  EXPECT_TRUE(result.has_value());

  EXPECT_DOUBLE_EQ(*result, 15.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DoubleCuBLASVectorOperationsTests, GetNormGetsNormForDoubleDenseVector)
{
  const vector<double> h_a {10.0, 10.0, 3.0, 4.0};
  DoubleDenseVector X {h_a.size()};

  X.copy_host_input_to_device(h_a);

  DoubleCuBLASVectorOperations operations {};

  const auto result = operations.get_norm(X);

  EXPECT_TRUE(result.has_value());

  EXPECT_DOUBLE_EQ(*result, 15.0);
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra
} // namespace GoogleUnitTests