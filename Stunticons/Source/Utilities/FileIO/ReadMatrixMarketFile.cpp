#include "FilePath.h"
#include "ReadMatrixMarketFile.h"

#include <iostream> // std::cerr
#include <sstream> // std::istringstream
#include <string> // std::getline
#include <vector>

using std::getline;
using std::istringstream;
using std::string;
using std::vector;

namespace Utilities
{
namespace FileIO
{

ReadMatrixMarketFile::ReadMatrixMarketFile(const FilePath& file_path):
  file_{file_path.file_path_},
  comments_{},
  number_of_rows_{0},
  number_of_columns_{0},
  number_of_nonzero_entries_{0},
  is_matrix_size_read_{false}
{
  if (!file_.is_open())
  {
    throw std::runtime_error(
      "Error opening file: " + file_path.file_path_.string());
  }
}

ReadMatrixMarketFile::~ReadMatrixMarketFile()
{
  file_.close();
}

vector<ReadMatrixMarketFile::MatrixMarketEntry>
  ReadMatrixMarketFile::read_file()
{
  string line {};

  getline(file_, line);

  // Read the first line.
  if (line != "%%MatrixMarket matrix coordinate real symmetric")
  {
    std::cerr << "Invalid MatrixMarket file format: " << std::endl;
  }

  while (!is_matrix_size_read_ && getline(file_, line))
  {
    if (line.empty())
    {
      continue;
    }
    // It's a comment.
    else if (line[0] == '%')
    {
      comments_.emplace_back(line);
    }
    // Read matrix size from the first, "non-comment" line.
    else
    {
      istringstream size_stream {line};
      size_stream >> number_of_rows_>> number_of_columns_ >>
        number_of_nonzero_entries_;

      is_matrix_size_read_ = true;
    }
  }

  vector<MatrixMarketEntry> matrix_entries {};
  matrix_entries.reserve(number_of_nonzero_entries_);

  while (getline(file_, line))
  {
    if (line.empty() || line[0] == '%')
    {
      // Skip empty lines and comment lines
      continue;
    }

    istringstream entry_stream {line};
    MatrixMarketEntry entry;

    entry_stream >> entry.row_ >> entry.column_ >> entry.value_;
    // Convert from 1-indexed to 0-indexed.
    entry.row_--;
    // Convert from 1-indexed to 0-indexed.
    entry.column_--;

    matrix_entries.emplace_back(entry);
  }

  return matrix_entries;
}

} // namespace FileIO
} // namespace Utilities
