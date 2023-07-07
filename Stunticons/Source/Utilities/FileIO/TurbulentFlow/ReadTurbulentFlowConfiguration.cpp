#include "ReadTurbulentFlowConfiguration.h"
#include "TurbulentFlowConfiguration.h"
#include "Utilities/FileIO/FilePath.h"

#include <regex>
#include <stdexcept>
#include <string> // std::getline, std::stoull
#include <vector>

using std::getline;
using std::istringstream;
using std::string;
using std::vector;

namespace Utilities
{
namespace FileIO
{

ReadTurbulentFlowConfiguration::ReadTurbulentFlowConfiguration(
  const FilePath& file_path
  ):
  file_{file_path.file_path_},
  comments_{}
{
  if (!file_.is_open())
  {
    throw std::runtime_error(
      "Error opening file: " + file_path.file_path_.string());
  }
}

ReadTurbulentFlowConfiguration::~ReadTurbulentFlowConfiguration()
{
  file_.close();
}

ReadTurbulentFlowConfiguration::Output
  ReadTurbulentFlowConfiguration::read_file()
{
  Output output {};
  auto std_size_t_map =
    TurbulentFlowConfiguration::create_std_size_t_parameters_map(
      output.std_size_t_parameters_);

  auto int_type_map =
    TurbulentFlowConfiguration::create_int_type_parameters_map(
      output.int_type_parameters_);

  auto double_type_map =
    TurbulentFlowConfiguration::create_double_type_parameters_map(
      output.double_type_parameters_);

  string line {};

  string variable_name {};

  // Matches 1 or more spaces.
  const std::regex white_space_re {"\\s+"};

  while (getline(file_, line))
  {
    if (line[0] == '#')
    {
      comments_.emplace_back(line);
    }
    else if (line.empty())
    {
      // TODO: Is continue better?
      // ignore extracts and discards characters from input stream until and
      // including delim which is '\n' in this case.
      // file_.ignore(max_stream_size, '\n');
      continue;
    }
    else
    {
      vector<string> tokens {
        // int submatch = -1 as fourth argument indicates text between matches
        // i.e. tokens or words) should be returned, not the matches themselves.
        std::sregex_token_iterator(
          line.begin(),
          line.end(),
          white_space_re,
          -1),
        {}};

      if (std_size_t_map.find(tokens.at(0)) != std_size_t_map.end())
      {
        *std_size_t_map.at(tokens.at(0)) = std::stoull(tokens.at(1));
      }

      if (int_type_map.find(tokens.at(0)) != int_type_map.end())
      {
        *int_type_map.at(tokens.at(0)) = std::stoi(tokens.at(1));
      }

      if (double_type_map.find(tokens.at(0)) != double_type_map.end())
      {
        *double_type_map.at(tokens.at(0)) = std::stod(tokens.at(1));
      }

      // Compare returns 0 if both character sequences compare equivalently.
      // The default value for count in basic_string substr(size_type pos = 0,
      // size_type count = npos) is npos and so the returned substr is from pos
      // to total size of string.
      // https://en.cppreference.com/w/cpp/string/basic_string/substr

      if (tokens.at(0).compare(0, 9, "wall_temp_") == 0)
      {
        output.unordered_map_type_parameters_.wall_temperatures_.insert(
          {std::stoul(tokens.at(0).substr(9)), std::stod(tokens.at(1))});
      }

      if (tokens.at(0).compare(0, 9, "wall_vel_") == 0)
      {
        output.unordered_map_type_parameters_.wall_velocities_.insert(
          {std::stoul(tokens.at(0).substr(9)), std::stod(tokens.at(1))});
      }

      if (tokens.at(0).compare(0, 4, "UIN_") == 0)
      {
        output.unordered_map_type_parameters_.inlet_Us_.insert(
          {std::stoul(tokens.at(0).substr(4)), std::stod(tokens.at(1))});
      }

      if (tokens.at(0).compare(0, 4, "VIN_") == 0)
      {
        output.unordered_map_type_parameters_.inlet_Vs_.insert(
          {std::stoul(tokens.at(0).substr(4)), std::stod(tokens.at(1))});
      }

      if (tokens.at(0).compare(0, 4, "TIN_") == 0)
      {
        output.unordered_map_type_parameters_.inlet_Ts_.insert(
          {std::stoul(tokens.at(0).substr(4)), std::stod(tokens.at(1))});
      }

      if (tokens.at(0).compare(0, 4, "KIN_") == 0)
      {
        output.unordered_map_type_parameters_.inlet_Ks_.insert(
          {std::stoul(tokens.at(0).substr(4)), std::stod(tokens.at(1))});
      }

      if (tokens.at(0).compare(0, 6, "EPSIN_") == 0)
      {
        output.unordered_map_type_parameters_.inlet_eps_.insert(
          {std::stoul(tokens.at(0).substr(6)), std::stod(tokens.at(1))});
      }
    }
  }

  return output;
}

} // namespace FileIO
} // namespace Utilities
