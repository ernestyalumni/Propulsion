#include "Algebra/Modules/Matrices/CompressedSparseRow.h"
#include "gtest/gtest.h"

#include <cstddef>

using Algebra::Modules::Matrices::SparseMatrices::CompressedSparseRowMatrix;
using Algebra::Modules::Matrices::SparseMatrices::DenseVector;

namespace GoogleUnitTests
{
namespace Algebra
{
namespace Modules
{
namespace Matrices
{

namespace CompressedSparseRow
{

constexpr std::size_t M {1048576};
constexpr std::size_t N {1048576 + 1};

} // namespace CompressedSparseRow

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CompressedSparseRowTests, Constructible)
{
  constexpr std::size_t number_of_nonzero_elements {
    (CompressedSparseRow::M - 2) * 3 + 4};

  CompressedSparseRowMatrix csr {
    CompressedSparseRow::M,
    CompressedSparseRow::N,number_of_nonzero_elements};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CompressedSparseRowTests, Destructible)
{
  constexpr std::size_t number_of_nonzero_elements {
    (CompressedSparseRow::M - 2) * 3 + 4};

  {
    CompressedSparseRowMatrix csr {
      CompressedSparseRow::M,
      CompressedSparseRow::N,number_of_nonzero_elements};
  }

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DenseVectorTests, Constructible)
{
  DenseVector dense_vector {CompressedSparseRow::M};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DenseVectorTests, Destructible)
{
  {
    DenseVector dense_vector {CompressedSparseRow::M};
  }

  SUCCEED();
}

} // namespace Matrices
} // namespace Modules
} // namespace Algebra
} // namespace GoogleUnitTests