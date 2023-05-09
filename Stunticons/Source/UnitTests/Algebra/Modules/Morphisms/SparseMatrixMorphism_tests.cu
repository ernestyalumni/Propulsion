#include "Algebra/Modules/Matrices/GenerateCompressedSparseRowMatrix.h"
#include "Algebra/Modules/Matrices/HostCompressedSparseRow.h"
#include "Algebra/Modules/Morphisms/SparseMatrixMorphism.h"
#include "gtest/gtest.h"

#include <cstddef>

using Algebra::Modules::Matrices::SparseMatrices::HostCompressedSparseRowMatrix;
using std::size_t;

namespace GoogleUnitTests
{
namespace Algebra
{
namespace Modules
{
namespace Morphisms
{

namespace GenerateCompressedSparseRow
{

constexpr size_t M {1048576};
//constexpr size_t N {1048576 + 1};

} // namespace GenerateCompressedSparseRow

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SparseMatrixMorphismTests, HasCorrectIndiciesForSquareMatrix)
{
  constexpr size_t number_of_nonzero_elements {
    (GenerateCompressedSparseRow::M - 2) * 3 + 4};

  HostCompressedSparseRowMatrix h_csr {
    GenerateCompressedSparseRow::M,
    GenerateCompressedSparseRow::M,
    number_of_nonzero_elements};

  generate_tridiagonal_matrix(h_csr);

  // Now, essentially check the properties of a triagonal matrix.

  // The first row should have only 2 elements.
  EXPECT_EQ(h_csr.I_[0], 0);
  EXPECT_EQ(h_csr.J_[0], 0);
  EXPECT_EQ(h_csr.J_[1], 1);

  // The second row should now start to have 3 elements.

  EXPECT_EQ(h_csr.I_[1], 2);
  EXPECT_EQ(h_csr.J_[2], 0);
  EXPECT_EQ(h_csr.J_[3], 1);
  EXPECT_EQ(h_csr.J_[4], 2);

  std::size_t cumulative_sum_of_elements {5};
  for (std::size_t i {2}; i < GenerateCompressedSparseRow::M - 1; ++i)
  {
    EXPECT_EQ(h_csr.I_[i], cumulative_sum_of_elements);
    EXPECT_EQ(h_csr.J_[cumulative_sum_of_elements], i - 1);
    EXPECT_EQ(h_csr.J_[cumulative_sum_of_elements + 1], i);
    EXPECT_EQ(h_csr.J_[cumulative_sum_of_elements + 2], i + 1);

    cumulative_sum_of_elements += 3;
  }

  EXPECT_EQ(cumulative_sum_of_elements, h_csr.number_of_elements_ - 2);

  // There should only be 2 elements in the very last row, as the "upper"
  // diagonal gets "cut off."
  EXPECT_EQ(
    h_csr.I_[GenerateCompressedSparseRow::M - 1],
    h_csr.number_of_elements_ - 2);
  EXPECT_EQ(
    h_csr.I_[GenerateCompressedSparseRow::M],
    h_csr.number_of_elements_);

  // For the last 2 rows of the tridiagonal matrix:

  EXPECT_EQ(
    h_csr.J_[h_csr.number_of_elements_ - 5],
    GenerateCompressedSparseRow::M - 3);
  EXPECT_EQ(
    h_csr.J_[h_csr.number_of_elements_ - 4],
    GenerateCompressedSparseRow::M - 2);
  EXPECT_EQ(
    h_csr.J_[h_csr.number_of_elements_ - 3],
    GenerateCompressedSparseRow::M - 1);

  EXPECT_EQ(
    h_csr.J_[h_csr.number_of_elements_ - 2],
    GenerateCompressedSparseRow::M - 2);
  EXPECT_EQ(
    h_csr.J_[h_csr.number_of_elements_ - 1],
    GenerateCompressedSparseRow::M - 1);
}

} // namespace Morphisms
} // namespace Modules
} // namespace Algebra
} // namespace GoogleUnitTests