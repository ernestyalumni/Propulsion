#include "Manifolds/Euclidean/PgmGeometry/Grid2d.h"
#include "ParseGeometryFile.h"
#include "Utilities/FileIO/FilePath.h"

#include <cstddef>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <string> // std::getline

using Manifolds::Euclidean::PgmGeometry::Grid2d;
using Utilities::FileIO::FilePath;
using std::getline;
using std::size_t;
using std::string;

namespace Utilities
{
namespace FileIO
{
namespace TurbulentFlow
{

ParseGeometryFile::ParseGeometryFile(const FilePath& file_path):
  file_{file_path.file_path_},
  comments_{},
  depth_{}
{
  if (!file_.is_open())
  {
    throw std::runtime_error(
      "Error opening file: " + file_path.file_path_.string());
  }  
}

ParseGeometryFile::~ParseGeometryFile()
{
  file_.close();
}

Grid2d ParseGeometryFile::parse_geometry_file()
{
  string line {};

  // First line : version.
  getline(file_, line);
  if (line.compare("P2") != 0)
  {
    std::cerr << "First line of the PGM file should be P2" << std::endl;
  }

  // Second line : comment.
  getline(file_, line);
  comments_.emplace_back(line);

  std::stringstream ss {};

  // Continue with a stringstream.
  // rdbuf() returns a ptr to the associated std::strstreambuf.
  // << reads the entire contents of where the buffer is pointing to into the
  // stringstream.
  ss << file_.rdbuf();

  // Third line : size
  std::size_t M {};
  std::size_t N {};
  ss >> M >> N;

  // Fourth line : depth
  ss >> depth_;

  // Following lines : data.
  // Subtract 2 because Grid2d accounts for an additional cell(s) at the "ends."
  Grid2d grid {M - 2, N - 2};

  for (size_t backwards_j_plus {N}; backwards_j_plus > 0; --backwards_j_plus)
  {
    for (size_t i {0}; i < M; ++i)
    {
      ss >> grid.at(i, backwards_j_plus - 1);
    }
  }

  return grid;
}

} // namespace TurbulentFlow
} // namespace FileIO
} // namespace Utilities