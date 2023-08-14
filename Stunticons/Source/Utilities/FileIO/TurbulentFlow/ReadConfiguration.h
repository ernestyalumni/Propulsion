#ifndef UTILITIES_FILEIO_TURBULENT_FLOW_READ_CONFIGURATION_H
#define UTILITIES_FILEIO_TURBULENT_FLOW_READ_CONFIGURATION_H

#include "Configuration.h"
#include "Manifolds/Euclidean/PgmGeometry/Grid2dMetricData.h"
#include "Utilities/FileIO/FilePath.h"

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace Utilities
{
namespace FileIO
{
namespace TurbulentFlow
{

class ReadConfiguration
{
  public:

    struct Output
    {
      TurbulentFlow::Configuration::StdSizeTParameters std_size_t_parameters_;
      TurbulentFlow::Configuration::IntTypeParameters int_type_parameters_;
      TurbulentFlow::Configuration::DoubleTypeParameters
        double_type_parameters_;
      TurbulentFlow::Configuration::UnorderedMapTypeParameters
        unordered_map_type_parameters_;
    };

    ReadConfiguration(const FilePath& file_path);

    ~ReadConfiguration();

    //--------------------------------------------------------------------------
    /// \returns Attempt to read the file into a stream.
    //--------------------------------------------------------------------------
    Output read_file();

    inline bool is_file_open() const
    {
      return static_cast<bool>(file_);
    }

    static Manifolds::Euclidean::PgmGeometry::Grid2dMetricData
      parse_grid2d_metric_data(const Output& read_output);

    std::vector<std::string> comments_;

  private:

    std::ifstream file_;
};

} // namespace TurbulentFlow
} // namespace FileIO
} // namespace Utilities

#endif // UTILITIES_FILEIO_TURBULENT_FLOW_READ_CONFIGURATION_H