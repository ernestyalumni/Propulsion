#ifndef UTILITIES_FILEIO_TURBULENT_FLOW_PARSE_GEOMETRY_FILE_H
#define UTILITIES_FILEIO_TURBULENT_FLOW_PARSE_GEOMETRY_FILE_H

#include "Utilities/FileIO/FilePath.h"
#include "Manifolds/Euclidean/PgmGeometry/Grid2d.h"

#include <fstream>

namespace Utilities
{
namespace FileIO
{
namespace TurbulentFlow
{

class ParseGeometryFile
{
  public:

    ParseGeometryFile(const FilePath& file_path);

    ~ParseGeometryFile();

    //--------------------------------------------------------------------------
    /// \brief Read and parse PGM geometry file.
    /// \details From https://github.com/yuphin/turbulent-cfd/blob/master/src/Utilities.cpp#L30
    /// and example .pgm files, such as
    /// https://github.com/yuphin/turbulent-cfd/blob/master/example_cases/StepFlowTurb/StepFlow.pgm#L3
    /// if we're given in the 3rd. line
    /// M = number of rows, N = number columns
    /// in that order, then the grid is shown "rotated" by 90 degrees.
    //--------------------------------------------------------------------------
    Manifolds::Euclidean::PgmGeometry::Grid2d parse_geometry_file();

    std::vector<std::string> comments_;

    int depth_;

  private:

    std::ifstream file_;
};

} // namespace TurbulentFlow
} // namespace FileIO
} // namespace Utilities

#endif // UTILITIES_FILEIO_TURBULENT_FLOW_PARSE_GEOMETRY_FILE_H