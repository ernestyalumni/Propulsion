#ifndef UTILITIES_FILEIO_READ_MATRIX_MARKET_FILE_H
#define UTILITIES_FILEIO_READ_MATRIX_MARKET_FILE_H

#include "Algebra/Modules/Matrices/HostCompressedSparseRow.h"

#include "FilePath.h"

#include <cstddef> // std::size_t
#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace Utilities
{
namespace FileIO
{

class ReadMatrixMarketFile
{
  public:

    struct MatrixMarketEntry
    {
      int row_;
      int column_;
      double value_;
    };

    ReadMatrixMarketFile(const FilePath& file_path);

    ~ReadMatrixMarketFile();

    std::vector<MatrixMarketEntry> read_file();

    //--------------------------------------------------------------------------
    /// \returns Tuple of row offsets, column indices, and non-zero values.
    //--------------------------------------------------------------------------
    std::tuple<std::vector<int>, std::vector<int>, std::vector<double>>
      read_file_as_compressed_sparse_row();

    inline bool is_file_open() const
    {
      return static_cast<bool>(file_);
    }

    Algebra::Modules::Matrices::SparseMatrices::
      DoubleHostCompressedSparseRowMatrix read_into_csr();

    Algebra::Modules::Matrices::SparseMatrices::HostCompressedSparseRowMatrix
      read_into_float_csr();

    std::vector<std::string> comments_;

    std::size_t number_of_rows_;
    std::size_t number_of_columns_;
    std::size_t number_of_nonzero_entries_;

  protected:

    static std::vector<std::map<int, double>> convert_entries_into_rows(
      const std::size_t number_of_rows,
      const std::size_t number_of_nonzero_entries,
      std::vector<MatrixMarketEntry>& matrix_entries);

    static void sort_by_row(std::vector<MatrixMarketEntry>& matrix_rows);

    //--------------------------------------------------------------------------
    /// \returns Tuple of row offsets, column indices, and non-zero values.
    //--------------------------------------------------------------------------
    static std::tuple<std::vector<int>, std::vector<int>, std::vector<double>>
      convert_entries_into_compressed_sparse_row(
      const std::size_t number_of_rows,
      const std::size_t number_of_nonzero_entries,
      std::vector<MatrixMarketEntry>& matrix_entries);

    //--------------------------------------------------------------------------
    /// \returns Tuple of row offsets, column indices, and non-zero values.
    //--------------------------------------------------------------------------
    static std::tuple<std::vector<int>, std::vector<int>, std::vector<double>>
      convert_rows_into_compressed_sparse_row(
      const std::size_t number_of_rows,
      const std::size_t number_of_nonzero_entries,
      std::vector<std::map<int, double>>& matrix_rows);

  private:

    std::ifstream file_;

    bool is_matrix_size_read_;
};

class ReadColumnVectorMarketFile
{
  public:

    ReadColumnVectorMarketFile(const FilePath& file_path);

    ~ReadColumnVectorMarketFile();

    std::vector<double> read_file();

    std::vector<float> read_file_as_float();

    std::vector<std::string> comments_;

    std::size_t number_of_rows_;
    std::size_t number_of_columns_;

  private:

    std::ifstream file_;

    bool is_matrix_size_read_;
};

} // namespace FileIO
} // namespace Utilities

#endif // UTILITIES_FILEIO_READ_MATRIX_MARKET_FILE_H