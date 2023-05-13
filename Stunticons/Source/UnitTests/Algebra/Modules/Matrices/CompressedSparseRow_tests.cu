#include "Algebra/Modules/Matrices/CompressedSparseRow.h"
#include "Algebra/Modules/Matrices/GenerateCompressedSparseRowMatrix.h"
#include "Algebra/Modules/Matrices/HostCompressedSparseRow.h"
#include "Algebra/Modules/Vectors/HostArrays.h"
#include "gtest/gtest.h"

#include <cstddef>

using Algebra::Modules::Matrices::SparseMatrices::CompressedSparseRowMatrix;
using Algebra::Modules::Matrices::SparseMatrices::DenseVector;
using Algebra::Modules::Matrices::SparseMatrices::generate_tridiagonal_matrix;
using Algebra::Modules::Matrices::SparseMatrices::HostCompressedSparseRowMatrix;
using Algebra::Modules::Vectors::HostArray;
using std::size_t;

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

static constexpr size_t M {1048576};
static constexpr size_t N {1048576 + 1};

} // namespace CompressedSparseRow

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CompressedSparseRowTests, Constructible)
{
  constexpr size_t number_of_nonzero_elements {
    (CompressedSparseRow::M - 2) * 3 + 4};

  CompressedSparseRowMatrix csr {
    CompressedSparseRow::M,
    CompressedSparseRow::N,
    number_of_nonzero_elements};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CompressedSparseRowTests, Destructible)
{
  constexpr size_t number_of_nonzero_elements {
    (CompressedSparseRow::M - 2) * 3 + 4};

  {
    CompressedSparseRowMatrix csr {
      CompressedSparseRow::M,
      CompressedSparseRow::N,
      number_of_nonzero_elements};
  }

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CompressedSparseRowTests, CopiesToDevice)
{
  constexpr size_t number_of_nonzero_elements {
    (CompressedSparseRow::M - 2) * 3 + 4};

  HostCompressedSparseRowMatrix h_csr {
    CompressedSparseRow::M,
    CompressedSparseRow::N,
    number_of_nonzero_elements};

  generate_tridiagonal_matrix(h_csr);

  CompressedSparseRowMatrix csr {
    CompressedSparseRow::M,
    CompressedSparseRow::N,
    number_of_nonzero_elements};

  csr.copy_host_input_to_device(h_csr);

  HostCompressedSparseRowMatrix h_output {
    CompressedSparseRow::M,
    CompressedSparseRow::N,
    number_of_nonzero_elements};

  csr.copy_device_output_to_host(h_output);

  for (size_t i {0}; i < h_csr.number_of_elements_; ++i)
  {
    EXPECT_FLOAT_EQ(h_csr.values_[i], h_output.values_[i]);
    EXPECT_EQ(h_csr.J_[i], h_output.J_[i]);
  }

  for (size_t i {0}; i < h_csr.M_ + 1; ++i)
  {
    EXPECT_EQ(h_csr.I_[i], h_output.I_[i]);
  }
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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DenseVectorTests, CopiesToDevice)
{
  HostArray rhs {CompressedSparseRow::M};
  HostArray x {CompressedSparseRow::M};

  for (size_t i {0}; i < CompressedSparseRow::M; ++i)
  {
    rhs.values_[i] = 1.0;
    x.values_[i] = 0.0;
  }

  DenseVector d_x {CompressedSparseRow::M};
  DenseVector d_r {CompressedSparseRow::M};

  d_x.copy_host_input_to_device(x);
  d_r.copy_host_input_to_device(rhs);

  HostArray rhs_out {CompressedSparseRow::M};
  HostArray x_out {CompressedSparseRow::M};

  for (size_t i {0}; i < CompressedSparseRow::M; ++i)
  {
    rhs_out.values_[i] = 42.0;
    x_out.values_[i] = 69.0;
  }

  d_x.copy_device_output_to_host(x_out);
  d_r.copy_device_output_to_host(rhs_out);

  for (size_t i {0}; i < CompressedSparseRow::M; ++i)
  {
    EXPECT_FLOAT_EQ(rhs_out.values_[i], 1.0);
    EXPECT_FLOAT_EQ(x_out.values_[i], 0.0);
  }
}

} // namespace Matrices
} // namespace Modules
} // namespace Algebra
} // namespace GoogleUnitTests