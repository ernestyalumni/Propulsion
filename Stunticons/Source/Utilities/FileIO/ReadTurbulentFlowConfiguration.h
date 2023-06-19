#ifndef UTILITIES_FILEIO_READ_TURBULENT_FLOW_CONFIGURATION_H
#define UTILITIES_FILEIO_READ_TURBULENT_FLOW_CONFIGURATION_H

#include "FilePath.h"
#include "TurbulentFlowConfiguration.h"

#include <fstream>
#include <string>
#include <vector>

namespace Utilities
{
namespace FileIO
{

class ReadTurbulentFlowConfiguration
{
  public:

    struct Output
    {
      TurbulentFlowConfiguration::StdSizeTParameters std_size_t_parameters_;
      TurbulentFlowConfiguration::IntTypeParameters int_type_parameters_;
      TurbulentFlowConfiguration::DoubleTypeParameters double_type_parameters_;
    };

    ReadTurbulentFlowConfiguration(const FilePath& file_path);

    ~ReadTurbulentFlowConfiguration();

    //--------------------------------------------------------------------------
    /// \returns Attempt to read the file into a stream.
    //--------------------------------------------------------------------------
    Output read_file();

    inline bool is_file_open() const
    {
      return static_cast<bool>(file_);
    }

    std::vector<std::string> comments_;

  private:

    std::ifstream file_;
};

} // namespace FileIO
} // namespace Utilities

#endif // UTILITIES_FILEIO_READ_TURBULENT_FLOW_CONFIGURATION_H