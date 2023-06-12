#include "PerformanceTests/Algebra/Solvers/SetupBiconjugateGradientStabilizedWithEigen.h"
#include "Utilities/FileIO/FilePath.h"
#include "Utilities/FileIO/ReadMatrixMarketFile.h"
#include "gtest/gtest.h"

#include <tuple> // std::get

using SetupWithEigen =
  PerformanceTests::Algebra::Solvers::
    SetupBiconjugateGradientStabilizedWithEigen;
using Utilities::FileIO::FilePath;
using Utilities::FileIO::ReadColumnVectorMarketFile;
using Utilities::FileIO::ReadMatrixMarketFile;
using std::get;

namespace GoogleUnitTests
{
namespace PerformanceTests
{
namespace Algebra
{
namespace Solvers
{

static const std::string relative_sparse_matrix_example_path_1 {
  "SuiteSparseMatrixCollection/1537_OptimizationProblem/c-18"};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetupBiconjugateGradientStabilizedWithEigenTests, Constructible)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_sparse_matrix_example_path_1);
  fp.append("c-18.mtx");
  ReadMatrixMarketFile read_mtx {fp};

  read_mtx.read_file_as_compressed_sparse_row();

  SetupWithEigen setup {read_mtx.number_of_rows_, read_mtx.number_of_columns_};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetupBiconjugateGradientStabilizedWithEigenTests, InsertIntoAInserts)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_sparse_matrix_example_path_1);
  fp.append("c-18.mtx");
  ReadMatrixMarketFile read_mtx {fp};

  const auto host_csr = read_mtx.read_file_as_compressed_sparse_row();

  SetupWithEigen setup {read_mtx.number_of_rows_, read_mtx.number_of_columns_};

  setup.insert_into_A(get<0>(host_csr), get<1>(host_csr), get<2>(host_csr));

  EXPECT_EQ(setup.A_.rows(), 2169);
  EXPECT_EQ(setup.A_.cols(), 2169);
  EXPECT_EQ(setup.A_.size(), 2169 * 2169);
  EXPECT_DOUBLE_EQ(setup.A_.coeff(0, 0), 3.7515151744965491);
  EXPECT_DOUBLE_EQ(setup.A_.coeff(1274, 0), 0.0199011749268386);
  EXPECT_DOUBLE_EQ(setup.A_.coeff(1275, 0), 0.04833602575091756);
  EXPECT_DOUBLE_EQ(setup.A_.coeff(2168, 2168), -1e-8);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetupBiconjugateGradientStabilizedWithEigenTests, InsertIntobInserts)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_sparse_matrix_example_path_1);
  fp.append("c-18_b.mtx");
  ReadColumnVectorMarketFile read_mtxb {fp};
  const auto host_b = read_mtxb.read_file();

  SetupWithEigen setup {host_b.size(), host_b.size()};

  setup.insert_into_b(host_b);

  EXPECT_EQ(setup.b_.rows(), 2169);
  EXPECT_EQ(setup.b_.cols(), 1);
  EXPECT_EQ(setup.b_.size(), 2169);
  EXPECT_DOUBLE_EQ(setup.b_[0], 7.8035053216857428e-05);
  EXPECT_DOUBLE_EQ(setup.b_[1], 0.0001182290988454583);
  EXPECT_DOUBLE_EQ(setup.b_[2168], 0);
}

} // namespace Solvers
} // namespace Algebra
} // namespace PerformanceTests
} // namespace GoogleUnitTests  
