#include "Algebra/Modules/Matrices/HostCompressedSparseRow.h"
#include "Utilities/FileIO/FilePath.h"
#include "Utilities/FileIO/ReadMatrixMarketFile.h"
#include "gtest/gtest.h"

#include <string>
#include <tuple> // std::get
#include <vector>

using HostCSR =
  Algebra::Modules::Matrices::SparseMatrices::
    DoubleHostCompressedSparseRowMatrix;
using Utilities::FileIO::FilePath;
using Utilities::FileIO::ReadColumnVectorMarketFile;
using Utilities::FileIO::ReadMatrixMarketFile;
using std::get;

using MatrixMarketEntry = ReadMatrixMarketFile::MatrixMarketEntry;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace FileIO
{

static const std::string relative_sparse_matrix_example_path_1 {
  "SuiteSparseMatrixCollection/1537_OptimizationProblem/c-18"};

class TestableReadMatrixMarketFile : public ReadMatrixMarketFile
{
  public:

    using ReadMatrixMarketFile::ReadMatrixMarketFile;
    using ReadMatrixMarketFile::sort_by_row;
    using ReadMatrixMarketFile::convert_entries_into_rows;
    using ReadMatrixMarketFile::convert_entries_into_compressed_sparse_row;
    using ReadMatrixMarketFile::convert_rows_into_compressed_sparse_row;
    using ReadMatrixMarketFile::number_of_rows_;
    using ReadMatrixMarketFile::number_of_columns_;
};

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
TEST(ReadMatrixMarketFileTests, ReadFileReadsFile)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_sparse_matrix_example_path_1);
  fp.append("c-18.mtx");
  TestableReadMatrixMarketFile read_mtx {fp};

  const auto matrix_result = read_mtx.read_file();

  EXPECT_EQ(read_mtx.number_of_rows_, 2169);
  EXPECT_EQ(read_mtx.number_of_columns_, 2169);
  EXPECT_EQ(read_mtx.number_of_nonzero_entries_, 8657);

  EXPECT_EQ(matrix_result.size(), 8657);

  EXPECT_EQ(matrix_result.at(0).row_, 0);
  EXPECT_EQ(matrix_result.at(0).column_, 0);
  EXPECT_DOUBLE_EQ(matrix_result.at(0).value_, 3.751515174496549);

  EXPECT_EQ(matrix_result.at(1).row_, 1274);
  EXPECT_EQ(matrix_result.at(1).column_, 0);
  EXPECT_DOUBLE_EQ(matrix_result.at(1).value_, 0.0199011749268386);

  EXPECT_EQ(matrix_result.at(8656).row_, 2168);
  EXPECT_EQ(matrix_result.at(8656).column_, 2168);
  EXPECT_DOUBLE_EQ(matrix_result.at(8656).value_, -1e-8);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadMatrixMarketFileTests, ReadFileAsCompressedSparseRowReads)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_sparse_matrix_example_path_1);
  fp.append("c-18.mtx");
  ReadMatrixMarketFile read_mtx {fp};

  const auto matrix_result = read_mtx.read_file_as_compressed_sparse_row();

  EXPECT_EQ(get<0>(matrix_result).size(), 2170);
  EXPECT_EQ(get<1>(matrix_result).size(), 8657);
  EXPECT_EQ(get<2>(matrix_result).size(), 8657);

  EXPECT_EQ(get<0>(matrix_result).at(0), 0);
  EXPECT_EQ(get<0>(matrix_result).at(1), 1);
  EXPECT_EQ(get<0>(matrix_result).at(2168), 8487);
  EXPECT_EQ(get<0>(matrix_result).at(2169), 8657);

  EXPECT_EQ(get<1>(matrix_result).at(0), 0);
  EXPECT_EQ(get<1>(matrix_result).at(1), 1);
  EXPECT_EQ(get<1>(matrix_result).at(8655), 1267);
  EXPECT_EQ(get<1>(matrix_result).at(8656), 2168);

  EXPECT_DOUBLE_EQ(get<2>(matrix_result).at(0), 3.7515151744965491);
  EXPECT_DOUBLE_EQ(get<2>(matrix_result).at(1), 3.7477497575623362);
  EXPECT_DOUBLE_EQ(get<2>(matrix_result).at(8655), 1);
  EXPECT_DOUBLE_EQ(get<2>(matrix_result).at(8656), -1e-08);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(
  ReadMatrixMarketFileTests,
  CanBeReadIntoDoubleHostCompressedSparseRowMatrix)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_sparse_matrix_example_path_1);
  fp.append("c-18.mtx");
  ReadMatrixMarketFile read_mtx {fp};

  const auto matrix_result = read_mtx.read_file_as_compressed_sparse_row();

  HostCSR csr {
    read_mtx.number_of_rows_,
    read_mtx.number_of_columns_,
    read_mtx.number_of_nonzero_entries_};
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadMatrixMarketFileTests, SortByRowSorts)
{
  std::vector<MatrixMarketEntry> example {};
  example.emplace_back(MatrixMarketEntry{2, 2, 5});
  example.emplace_back(MatrixMarketEntry{0, 0, 1});
  example.emplace_back(MatrixMarketEntry{1, 2, 3});
  example.emplace_back(MatrixMarketEntry{2, 0, 4});
  example.emplace_back(MatrixMarketEntry{0, 1, 2});

  TestableReadMatrixMarketFile::sort_by_row(example);

  EXPECT_EQ(example.at(0).row_, 0);
  EXPECT_EQ(example.at(1).row_, 0);
  EXPECT_EQ(example.at(2).row_, 1);
  EXPECT_EQ(example.at(3).row_, 2);
  EXPECT_EQ(example.at(4).row_, 2);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadMatrixMarketFileTests, ConvertEntriesIntoCompressedSparseRowConverts)
{
  std::vector<MatrixMarketEntry> example {};
  example.emplace_back(MatrixMarketEntry{0, 0, 1});
  example.emplace_back(MatrixMarketEntry{0, 1, 2});
  example.emplace_back(MatrixMarketEntry{1, 2, 3});
  example.emplace_back(MatrixMarketEntry{2, 0, 4});
  example.emplace_back(MatrixMarketEntry{2, 2, 5});

  const auto result =
    TestableReadMatrixMarketFile::convert_entries_into_compressed_sparse_row(
      3,
      5,
      example);

  EXPECT_EQ(get<0>(result).at(0), 0);
  EXPECT_EQ(get<0>(result).at(1), 2);
  EXPECT_EQ(get<0>(result).at(2), 3);
  EXPECT_EQ(get<0>(result).at(3), 5);

  EXPECT_EQ(get<1>(result).at(0), 0);
  EXPECT_EQ(get<1>(result).at(1), 1);
  EXPECT_EQ(get<1>(result).at(2), 2);
  EXPECT_EQ(get<1>(result).at(3), 0);
  EXPECT_EQ(get<1>(result).at(4), 2);

  EXPECT_DOUBLE_EQ(get<2>(result).at(0), 1);
  EXPECT_DOUBLE_EQ(get<2>(result).at(1), 2);
  EXPECT_DOUBLE_EQ(get<2>(result).at(2), 3);
  EXPECT_DOUBLE_EQ(get<2>(result).at(3), 4);
  EXPECT_DOUBLE_EQ(get<2>(result).at(4), 5);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadMatrixMarketFileTests, ConvertEntriesIntoRowsConverts)
{
  std::vector<MatrixMarketEntry> example {};
  example.emplace_back(MatrixMarketEntry{0, 0, 1});
  example.emplace_back(MatrixMarketEntry{0, 1, 2});
  example.emplace_back(MatrixMarketEntry{1, 2, 3});
  example.emplace_back(MatrixMarketEntry{2, 0, 4});
  example.emplace_back(MatrixMarketEntry{2, 2, 5});

  const auto result =
    TestableReadMatrixMarketFile::convert_entries_into_rows(
      3,
      5,
      example);

  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result.at(0).size(), 2);
  EXPECT_EQ(result.at(1).size(), 1);
  EXPECT_EQ(result.at(2).size(), 2);

  EXPECT_DOUBLE_EQ(result.at(0).at(0), 1);
  EXPECT_DOUBLE_EQ(result.at(0).at(1), 2);
  EXPECT_DOUBLE_EQ(result.at(1).at(2), 3);
  EXPECT_DOUBLE_EQ(result.at(2).at(0), 4);
  EXPECT_DOUBLE_EQ(result.at(2).at(2), 5);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadMatrixMarketFileTests, ConvertRowsIntoCompressedSparseRowConverts)
{
  std::vector<MatrixMarketEntry> example {};
  example.emplace_back(MatrixMarketEntry{0, 0, 1});
  example.emplace_back(MatrixMarketEntry{0, 1, 2});
  example.emplace_back(MatrixMarketEntry{1, 2, 3});
  example.emplace_back(MatrixMarketEntry{2, 0, 4});
  example.emplace_back(MatrixMarketEntry{2, 2, 5});

  auto example_as_rows =
    TestableReadMatrixMarketFile::convert_entries_into_rows(
      3,
      5,
      example);

  const auto result =
   TestableReadMatrixMarketFile::convert_rows_into_compressed_sparse_row(
      3,
      5,
      example_as_rows);

  EXPECT_EQ(get<0>(result).at(0), 0);
  EXPECT_EQ(get<0>(result).at(1), 2);
  EXPECT_EQ(get<0>(result).at(2), 3);
  EXPECT_EQ(get<0>(result).at(3), 5);

  EXPECT_EQ(get<1>(result).at(0), 0);
  EXPECT_EQ(get<1>(result).at(1), 1);
  EXPECT_EQ(get<1>(result).at(2), 2);
  EXPECT_EQ(get<1>(result).at(3), 0);
  EXPECT_EQ(get<1>(result).at(4), 2);

  EXPECT_DOUBLE_EQ(get<2>(result).at(0), 1);
  EXPECT_DOUBLE_EQ(get<2>(result).at(1), 2);
  EXPECT_DOUBLE_EQ(get<2>(result).at(2), 3);
  EXPECT_DOUBLE_EQ(get<2>(result).at(3), 4);
  EXPECT_DOUBLE_EQ(get<2>(result).at(4), 5);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadColumnVectorMarketFileTests, Constructible)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_sparse_matrix_example_path_1);
  fp.append("c-18_b.mtx");
  ReadColumnVectorMarketFile read_mtx {fp};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadColumnVectorMarketFileTests, Destructible)
{
  {
    FilePath fp {FilePath::get_data_directory()};
    fp.append(relative_sparse_matrix_example_path_1);
    fp.append("c-18_b.mtx");
    ReadColumnVectorMarketFile read_mtx {fp};
  }

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ReadColumnVectorMarketFileTests, ReadFileReadsFile)
{
  FilePath fp {FilePath::get_data_directory()};
  fp.append(relative_sparse_matrix_example_path_1);
  fp.append("c-18_b.mtx");
  ReadColumnVectorMarketFile read_mtx {fp};

  const auto matrix_result = read_mtx.read_file();

  EXPECT_EQ(read_mtx.number_of_rows_, 2169);
  EXPECT_EQ(read_mtx.number_of_columns_, 1);

  EXPECT_EQ(matrix_result.size(), 2169);

  EXPECT_DOUBLE_EQ(matrix_result.at(0), 7.8035053216857428e-05);

  EXPECT_DOUBLE_EQ(matrix_result.at(1), 0.0001182290988454583);

  EXPECT_DOUBLE_EQ(matrix_result.at(2168), 0.0);
}

} // namespace FileIO
} // namespace Utilities
} // namespace GoogleUnitTests