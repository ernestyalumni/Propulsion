#include "Algebra/Modules/Matrices/HostCompressedSparseRow.h"
#include "FilePath.h"
#include "ReadMatrixMarketFile.h"

#include <algorithm>
#include <cassert>
#include <cstddef> // std::size_t
#include <iostream> // std::cerr
#include <map>
#include <sstream> // std::istringstream
#include <string> // std::getline
#include <tuple> // std::get
#include <vector>

using std::get;
using std::getline;
using std::istringstream;
using std::size_t;
using std::string;
using std::tuple;
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

Algebra::Modules::Matrices::SparseMatrices::DoubleHostCompressedSparseRowMatrix
  ReadMatrixMarketFile::read_into_csr()
{
  const auto read_result = read_file_as_compressed_sparse_row();

  Algebra::Modules::Matrices::SparseMatrices::
    DoubleHostCompressedSparseRowMatrix csr {
      number_of_rows_,
      number_of_columns_,
      number_of_nonzero_entries_};

  csr.copy_row_offsets(get<0>(read_result));
  csr.copy_column_indices(get<1>(read_result));
  csr.copy_values(get<2>(read_result));

  return csr;
}

Algebra::Modules::Matrices::SparseMatrices::HostCompressedSparseRowMatrix
  ReadMatrixMarketFile::read_into_float_csr()
{
  const auto read_result = read_file_as_compressed_sparse_row();

  Algebra::Modules::Matrices::SparseMatrices::HostCompressedSparseRowMatrix
    csr {
      number_of_rows_,
      number_of_columns_,
      number_of_nonzero_entries_};

  const vector<float> values_as_floats {
    get<2>(read_result).begin(),
    get<2>(read_result).end()}; 

  csr.copy_row_offsets(get<0>(read_result));
  csr.copy_column_indices(get<1>(read_result));
  csr.copy_values(values_as_floats);

  return csr;
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

tuple<vector<int>, vector<int>, vector<double>>
  ReadMatrixMarketFile::read_file_as_compressed_sparse_row()
{
  auto matrix_entries = read_file();

  auto matrix_rows = ReadMatrixMarketFile::convert_entries_into_rows(
    number_of_rows_,
    number_of_nonzero_entries_,
    matrix_entries);

  return ReadMatrixMarketFile::convert_rows_into_compressed_sparse_row(
    number_of_rows_,
    number_of_nonzero_entries_,
    matrix_rows);
}


vector<std::map<int, double>> ReadMatrixMarketFile::convert_entries_into_rows(
  const size_t number_of_rows,
  const size_t number_of_nonzero_entries,
  vector<MatrixMarketEntry>& matrix_entries
  )
{
  ReadMatrixMarketFile::sort_by_row(matrix_entries);

  vector<std::map<int, double>> matrix_rows (number_of_rows);

  for (MatrixMarketEntry& entry : matrix_entries)
  {
    matrix_rows.at(entry.row_)[entry.column_] = entry.value_;
  }

  return matrix_rows;
}

void ReadMatrixMarketFile::sort_by_row(vector<MatrixMarketEntry>& matrix_rows)
{
  std::sort(
    matrix_rows.begin(),
    matrix_rows.end(),
    [](const MatrixMarketEntry& a, const MatrixMarketEntry& b) -> bool
    {
      return a.row_ < b.row_;
    });
}

tuple<vector<int>, vector<int>, vector<double>>
  ReadMatrixMarketFile::convert_entries_into_compressed_sparse_row(
  const size_t number_of_rows,
  const size_t number_of_nonzero_entries,
  vector<MatrixMarketEntry>& matrix_entries)
{
  ReadMatrixMarketFile::sort_by_row(matrix_entries);

  vector<int> row_offsets (number_of_rows + 1, 0);
  row_offsets.at(number_of_rows) = number_of_nonzero_entries;
  row_offsets.at(0) = 0;

  vector<int> column_indices (number_of_nonzero_entries, 0);
  vector<double> values (number_of_nonzero_entries, 0.0);

  size_t i {0};
  size_t current_row_index {0};
  size_t row_offset_index {1};
  int row_offset_total {0};
  for (MatrixMarketEntry& entry : matrix_entries)
  {
    column_indices.at(i) = entry.column_;
    values.at(i) = entry.value_;
    ++i;

    if (current_row_index == entry.row_)
    {
      row_offset_total += 1;
    }
    else
    {
      assert(current_row_index < entry.row_);

      row_offsets.at(row_offset_index) = row_offset_total;

      const size_t row_index_difference {entry.row_ - current_row_index};

      current_row_index += 1;
      row_offset_index += 1;

      for (size_t j {1}; j < row_index_difference; ++j)
      {
        row_offsets.at(current_row_index) = row_offset_total;
        current_row_index += 1;
        row_offset_index += 1;
      }

      row_offset_total += 1;
    }
  }

  return std::make_tuple(row_offsets, column_indices, values);
}

tuple<vector<int>, vector<int>, vector<double>>
  ReadMatrixMarketFile::convert_rows_into_compressed_sparse_row(
  const size_t number_of_rows,
  const size_t number_of_nonzero_entries,
  vector<std::map<int, double>>& matrix_rows)
{
  vector<int> row_offsets (number_of_rows + 1, 0);
  row_offsets.at(number_of_rows) = number_of_nonzero_entries;
  row_offsets.at(0) = 0;

  vector<int> column_indices (number_of_nonzero_entries, 0);
  vector<double> values (number_of_nonzero_entries, 0.0);

  size_t i {0};
  size_t row_offset_total {0};

  for (size_t row_index {0}; row_index < number_of_rows; ++row_index)
  {
    for (const auto& column_index_value_pair : matrix_rows.at(row_index))
    {
      column_indices.at(i) = column_index_value_pair.first;
      values.at(i) = column_index_value_pair.second;

      ++i;
    }

    row_offset_total += matrix_rows.at(row_index).size();

    row_offsets.at(row_index + 1) = row_offset_total;
  }

  return std::make_tuple(row_offsets, column_indices, values);
}

ReadColumnVectorMarketFile::ReadColumnVectorMarketFile(
  const FilePath& file_path
  ):
  file_{file_path.file_path_},
  comments_{},
  number_of_rows_{0},
  number_of_columns_{0},
  is_matrix_size_read_{false}
{
  if (!file_.is_open())
  {
    throw std::runtime_error(
      "Error opening file: " + file_path.file_path_.string());
  }
}

ReadColumnVectorMarketFile::~ReadColumnVectorMarketFile()
{
  file_.close();
}

vector<double> ReadColumnVectorMarketFile::read_file()
{
  string line {};

  getline(file_, line);

  // Read the first line.
  if (line != "%%MatrixMarket matrix array real general")
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
      size_stream >> number_of_rows_>> number_of_columns_;

      is_matrix_size_read_ = true;
    }
  }

  vector<double> matrix_entries {};
  matrix_entries.reserve(number_of_rows_);

  while (getline(file_, line))
  {
    if (line.empty() || line[0] == '%')
    {
      // Skip empty lines and comment lines
      continue;
    }

    istringstream entry_stream {line};

    double value;

    entry_stream >> value;

    matrix_entries.emplace_back(value);
  }

  return matrix_entries;
}

vector<float> ReadColumnVectorMarketFile::read_file_as_float()
{
  string line {};

  getline(file_, line);

  // Read the first line.
  if (line != "%%MatrixMarket matrix array real general")
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
      size_stream >> number_of_rows_>> number_of_columns_;

      is_matrix_size_read_ = true;
    }
  }

  vector<float> matrix_entries {};
  matrix_entries.reserve(number_of_rows_);

  while (getline(file_, line))
  {
    if (line.empty() || line[0] == '%')
    {
      // Skip empty lines and comment lines
      continue;
    }

    istringstream entry_stream {line};

    float value;

    entry_stream >> value;

    matrix_entries.emplace_back(value);
  }

  return matrix_entries;
}

} // namespace FileIO
} // namespace Utilities
