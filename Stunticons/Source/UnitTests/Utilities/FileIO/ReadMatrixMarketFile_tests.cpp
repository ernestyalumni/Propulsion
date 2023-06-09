#include "Utilities/FileIO/FilePath.h"
#include "Utilities/FileIO/ReadMatrixMarketFile.h"
#include "gtest/gtest.h"

#include <string>

using Utilities::FileIO::FilePath;
using Utilities::FileIO::ReadMatrixMarketFile;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace FileIO
{

static const std::string relative_sparse_matrix_example_path_1 {
  "SuiteSparseMatrixCollection/1537_OptimizationProblem/c-18"};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadMatrixMarketFileTests, Constructible)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_sparse_matrix_example_path_1);
  fp.append("c-18.mtx");
  ReadMatrixMarketFile read_mtx {fp};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadMatrixMarketFileTests, Destructible)
{
  {
    FilePath fp {FilePath::get_data_directory()};
    fp.append(relative_sparse_matrix_example_path_1);
    fp.append("c-18.mtx");
    ReadMatrixMarketFile read_mtx {fp};
  }

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadMatrixMarketFileTests, CallOperatorReadsFile)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_sparse_matrix_example_path_1);
  fp.append("c-18.mtx");
  ReadMatrixMarketFile read_mtx {fp};

  const auto matrix_result = read_mtx.read_file();
}

} // namespace FileIO
} // namespace Utilities
} // namespace GoogleUnitTests