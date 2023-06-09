#include "FilePath.h"

#include <filesystem>
#include <string>

namespace fs = std::filesystem;

namespace Utilities
{
namespace FileIO
{

FilePath::FilePath(const std::string& file_path):
  file_path_{file_path == "" ? fs::current_path() : fs::path{file_path}}
{}

} // namespace FileIO
} // namespace Utilities
