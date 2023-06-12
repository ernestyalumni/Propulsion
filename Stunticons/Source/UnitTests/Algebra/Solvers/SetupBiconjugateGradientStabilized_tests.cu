#include "Algebra/Solvers/BiconjugateGradientStabilized.h"
#include "Algebra/Solvers/SetupBiconjugateGradientStabilized.h"
#include "Utilities/FileIO/FilePath.h"
#include "Utilities/FileIO/ReadMatrixMarketFile.h"
#include "gtest/gtest.h"

#include <tuple> // std::get
#include <vector>

using BiconjugateGradientStabilized =
  Algebra::Solvers::BiconjugateGradientStabilized;
using SetupBiconjugateGradientStabilized =
  Algebra::Solvers::SetupBiconjugateGradientStabilized;
using Utilities::FileIO::FilePath;
using Utilities::FileIO::ReadColumnVectorMarketFile;
using Utilities::FileIO::ReadMatrixMarketFile;
using std::get;
using std::vector;

namespace GoogleUnitTests
{
namespace Algebra
{
namespace Solvers
{

static const std::string relative_sparse_matrix_example_path_1 {
  "SuiteSparseMatrixCollection/1537_OptimizationProblem/c-18"};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetupBiconjugateGradientStabilizedTests, Constructible)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_sparse_matrix_example_path_1);
  fp.append("c-18.mtx");
  ReadMatrixMarketFile read_mtx {fp};

  fp.remove_filename();
  fp.append("c-18_b.mtx");
  ReadColumnVectorMarketFile read_mtxb {fp};

  SetupBiconjugateGradientStabilized setup {
    read_mtx.read_into_csr(),
    read_mtxb.read_file()};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(
  SetupBiconjugateGradientStabilizedTests,
  RunsBiconjugateGradientStabilizedMethod)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_sparse_matrix_example_path_1);
  fp.append("c-18.mtx");
  ReadMatrixMarketFile read_mtx {fp};

  const auto host_csr = read_mtx.read_into_csr();

  fp.remove_filename();
  fp.append("c-18_b.mtx");
  ReadColumnVectorMarketFile read_mtxb {fp};
  const auto host_b = read_mtxb.read_file();

  SetupBiconjugateGradientStabilized setup {host_csr, host_b};

  BiconjugateGradientStabilized cg {
    setup.A_,
    setup.b_,
    setup.morphism_,
    setup.operations_,
    20000};

  cg.create_default_initial_guess(setup.x_);

  const auto result = cg.solve(
    setup.x_,
    setup.Ax_,
    setup.r_,
    setup.p_,
    setup.s_);

  EXPECT_TRUE(get<0>(result));
  EXPECT_EQ(get<1>(result), 20000);

  vector<double> h_x_output (setup.M_, 0.0);
  vector<double> y (setup.M_, 0.0);

  setup.x_.copy_device_output_to_host(h_x_output);

  host_csr.multiply(h_x_output, y);

  SUCCEED();
}

} // namespace Solvers
} // namespace Algebra
} // namespace GoogleUnitTests  
