#ifndef UTILITIES_FILEIO_READ_MATRIX_MARKET_FILE_H
#define UTILITIES_FILEIO_READ_MATRIX_MARKET_FILE_H

#include "FilePath.h"

#include <cstddef> // std::size_t
#include <fstream>
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

    std::ifstream file_;

    std::vector<std::string> comments_;

    std::size_t number_of_rows_;
    std::size_t number_of_columns_;
    std::size_t number_of_nonzero_entries_;

    bool is_matrix_size_read_;
};

} // namespace FileIO
} // namespace Utilities

#endif // UTILITIES_FILEIO_READ_MATRIX_MARKET_FILE_H