#include "Algebra/Modules/Matrices/CompressedSparseRow.h"
#include "Algebra/Modules/Matrices/HostCompressedSparseRow.h"
#include "Algebra/Modules/Morphisms/SparseMatrixMorphism.h"
#include "Algebra/Modules/Vectors/HostArrays.h"
#include "gtest/gtest.h"

#include <array>
#include <cstddef>

using Algebra::Modules::Matrices::SparseMatrices::CompressedSparseRowMatrix;
using Algebra::Modules::Matrices::SparseMatrices::DenseVector;
using Algebra::Modules::Matrices::SparseMatrices::HostCompressedSparseRowMatrix;
using Algebra::Modules::Morphisms::SparseMatrixMorphismOnDenseVector;
using Algebra::Modules::Vectors::HostArray;
using std::array;
using std::size_t;

namespace GoogleUnitTests
{
namespace Algebra
{
namespace Modules
{
namespace Morphisms
{

struct SparseMatrixMorphismTestSetup
{
  static constexpr size_t NNZ_ {9};
  static constexpr size_t M_ {4};

  CompressedSparseRowMatrix A_;
  DenseVector X_;
  DenseVector Y_;

  SparseMatrixMorphismTestSetup():
    A_{M_, M_, NNZ_},
    X_{M_},
    Y_{M_}
  {
    HostCompressedSparseRowMatrix h_A {M_, M_, NNZ_};
    const array<float, NNZ_> values {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    h_A.copy_values(values);

    const array<int, M_ + 1> row_offsets {0, 3, 4, 7, 9};
    h_A.copy_row_offsets(row_offsets);

    const array<int, NNZ_> column_indices {0, 2, 3, 1, 0, 2, 3, 1, 3};

    h_A.copy_column_indices(column_indices);

    A_.copy_host_input_to_device(h_A);

    const array<float, M_> h_x {1.0f, 2.0f, 3.0f, 4.0f};
    const array<float, M_> h_y {0.0f, 0.0f, 0.0f, 0.0f};

    X_.copy_host_input_to_device(h_x);
    Y_.copy_host_input_to_device(h_y);
  }

  ~SparseMatrixMorphismTestSetup() = default;
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SparseMatrixMorphismOnDenseVectorTests, Constructible)
{
  SparseMatrixMorphismOnDenseVector morphism {1.0f, 0.0f};

  EXPECT_FLOAT_EQ(morphism.get_alpha(), 1.0f);
  EXPECT_FLOAT_EQ(morphism.get_beta(), 0.0f);
  EXPECT_EQ(morphism.get_buffer_size(), 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SparseMatrixMorphismOnDenseVectorTests, Destructible)
{
  {
    SparseMatrixMorphismOnDenseVector morphism {1.0f, 0.0f};
  }

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SparseMatrixMorphismOnDenseVectorTests, BufferSizeBuffersSize)
{
  SparseMatrixMorphismOnDenseVector morphism {1.0f, 0.0f};

  SparseMatrixMorphismTestSetup setup {};

  EXPECT_TRUE(morphism.buffer_size(setup.A_, setup.X_, setup.Y_));

  EXPECT_EQ(morphism.get_buffer_size(), 8);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(
  SparseMatrixMorphismOnDenseVectorTests,
  LinearTransformBeforeBufferReturnsFalse)
{
  SparseMatrixMorphismOnDenseVector morphism {1.0f, 0.0f};

  SparseMatrixMorphismTestSetup setup {};

  EXPECT_FALSE(morphism.linear_transform(setup.A_, setup.X_, setup.Y_));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SparseMatrixMorphismOnDenseVectorTests, LinearTransformAppliesMatrix)
{
  SparseMatrixMorphismOnDenseVector morphism {1.0f, 0.0f};

  SparseMatrixMorphismTestSetup setup {};

  EXPECT_TRUE(morphism.buffer_size(setup.A_, setup.X_, setup.Y_));

  // [1, 0, 2, 3]   [1]   [19]
  // [0, 4, 0, 0]   [2]    [8]
  // [5, 0, 6, 7]   [3]   [51]
  // [0, 8, 0, 9] x [4] = [52]

  EXPECT_TRUE(morphism.linear_transform(setup.A_, setup.X_, setup.Y_));

  HostArray y_out {setup.M_};

  setup.Y_.copy_device_output_to_host(y_out);

  EXPECT_FLOAT_EQ(y_out.values_[0], 19.0);
  EXPECT_FLOAT_EQ(y_out.values_[1], 8.0);
  EXPECT_FLOAT_EQ(y_out.values_[2], 51.0);
  EXPECT_FLOAT_EQ(y_out.values_[3], 52.0);
}

} // namespace Morphisms
} // namespace Modules
} // namespace Algebra
} // namespace GoogleUnitTests