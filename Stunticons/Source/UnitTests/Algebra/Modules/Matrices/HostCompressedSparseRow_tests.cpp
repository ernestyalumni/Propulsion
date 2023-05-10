#include "Algebra/Modules/Matrices/HostCompressedSparseRow.h"
#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <vector>

using Algebra::Modules::Matrices::SparseMatrices::HostCompressedSparseRowMatrix;
using std::array;
using std::size_t;

namespace GoogleUnitTests
{
namespace Algebra
{
namespace Modules
{
namespace Matrices
{
namespace SparseMatrices
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HostCompressedSparseRowTests, Constructs)
{
  HostCompressedSparseRowMatrix X {4, 5, 9};

  EXPECT_EQ(X.M_, 4);
  EXPECT_EQ(X.N_, 5);
  EXPECT_EQ(X.number_of_elements_, 9);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HostCompressedSparseRowTests, CopiesValuesWithArrays)
{
  static constexpr size_t NNZ {9};

  HostCompressedSparseRowMatrix X {4, 4, NNZ};

  const array<float, NNZ> values {
    1.0f,
    2.0f,
    3.0f,
    4.0f,
    5.0f,
    6.0f,
    7.0f,
    8.0f,
    9.0f};

  const auto result = X.copy_values(values);

  for (int i {0}; i < X.number_of_elements_; ++i)
  {
    EXPECT_FLOAT_EQ(X.values_[i], static_cast<float>(i + 1));
  }

  EXPECT_EQ(result - X.values_, 9);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HostCompressedSparseRowTests, CopiesRowOffsetsWithVectors)
{
  HostCompressedSparseRowMatrix X {4, 4, 9};

  const std::vector<int> row_offsets {0, 3, 4, 7, 9};

  const auto result = X.copy_row_offsets(row_offsets);

  EXPECT_EQ(result - X.I_, 5);

  EXPECT_EQ(X.I_[0], 0);
  EXPECT_EQ(X.I_[1], 3);
  EXPECT_EQ(X.I_[2], 4);
  EXPECT_EQ(X.I_[3], 7);
  EXPECT_EQ(X.I_[4], 9);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HostCompressedSparseRowTests, CopiesColumnIndicesWithArrays)
{
  HostCompressedSparseRowMatrix X {4, 4, 9};

  const std::array<int, 9> column_indices {0, 2, 3, 1, 0, 2, 3, 1, 3};

  const auto result = X.copy_column_indices(column_indices);

  EXPECT_EQ(result - X.J_, 9);

  EXPECT_EQ(X.J_[0], 0);
  EXPECT_EQ(X.J_[1], 2);
  EXPECT_EQ(X.J_[2], 3);
  EXPECT_EQ(X.J_[3], 1);
  EXPECT_EQ(X.J_[4], 0);
  EXPECT_EQ(X.J_[5], 2);
  EXPECT_EQ(X.J_[6], 3);
  EXPECT_EQ(X.J_[7], 1);
  EXPECT_EQ(X.J_[8], 3);
}

} // namespace SparseMatrices
} // namespace Matrices
} // namespace Modules
} // namespace Algebra
} // namespace GoogleUnitTests