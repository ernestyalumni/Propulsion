#include "Utilities/FileIO/FilePath.h"
#include "gtest/gtest.h"

#include <filesystem>

namespace fs = std::filesystem;

using Utilities::FileIO::FilePath;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace FileIO
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FilePathTests, DefaultConstructible)
{
  FilePath fp {};
  EXPECT_TRUE(fp.exists());
  EXPECT_TRUE(fp.is_directory());
  EXPECT_FALSE(fp.is_file());
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FilePathTests, DefaultsToCurrentPath)
{
  FilePath fp {};

  EXPECT_EQ(fp.get_relative_path().string(), ".");
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FilePathTests, SourceDirectoryExists)
{
  EXPECT_TRUE(fs::exists(FilePath::get_source_directory()));
  EXPECT_TRUE(fs::is_directory(FilePath::get_source_directory()));
  EXPECT_FALSE(fs::is_regular_file(FilePath::get_source_directory()));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FilePathTests, DataDirectoryExists)
{
  EXPECT_TRUE(fs::exists(FilePath::get_data_directory()));
  EXPECT_TRUE(fs::is_directory(FilePath::get_data_directory()));
  EXPECT_FALSE(fs::is_regular_file(FilePath::get_data_directory()));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(FilePathTests, AppendWorks)
{
  {
    FilePath fp {FilePath::get_data_directory()};
    fp.append("SuiteSparseMatrixCollection/1537_OptimizationProblem");
    EXPECT_EQ(
      fp.get_relative_path(),
      "../../Data/SuiteSparseMatrixCollection/1537_OptimizationProblem");
  }
  {
    FilePath fp {FilePath::get_data_directory()};
    fp.append("SuiteSparseMatrixCollection");
    EXPECT_EQ(
      fp.get_relative_path(),
      "../../Data/SuiteSparseMatrixCollection");
    fp.append("1537_OptimizationProblem");
    EXPECT_EQ(
      fp.get_relative_path(),
      "../../Data/SuiteSparseMatrixCollection/1537_OptimizationProblem");
  }
}

} // namespace FileIO
} // namespace Utilities
} // namespace GoogleUnitTests