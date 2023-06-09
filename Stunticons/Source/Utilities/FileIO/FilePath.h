#ifndef UTILITIES_FILEIO_FILE_PATH_H
#define UTILITIES_FILEIO_FILE_PATH_H

#include <filesystem>
#include <string>

namespace Utilities
{
namespace FileIO
{

class FilePath
{
  public:

    //--------------------------------------------------------------------------
    /// \details Defaults to the current path of execution.
    //--------------------------------------------------------------------------
    FilePath(const std::string& file_path = "");

    ~FilePath() = default;

    //--------------------------------------------------------------------------
    /// \details Appends elements to the path with a directory separator.
    //--------------------------------------------------------------------------

    inline void append(const std::string& path)
    {
      file_path_.append(path);
    }

    inline bool exists()
    {
      return std::filesystem::exists(file_path_);
    }

    inline bool is_directory()
    {
      return std::filesystem::is_directory(file_path_);
    }

    inline bool is_file()
    {
      return std::filesystem::is_regular_file(file_path_);
    }

    static inline std::filesystem::path get_source_directory()
    {
      return std::filesystem::path(__FILE__).parent_path().parent_path()
        .parent_path();
    }

    static inline std::filesystem::path get_data_directory()
    {
      return get_source_directory().parent_path().parent_path().concat("/Data");
    }

    inline std::filesystem::path get_absolute_path()
    {
      return std::filesystem::absolute(file_path_);
    }

    inline std::filesystem::path get_relative_path()
    {
      return std::filesystem::relative(file_path_);
    }

    std::filesystem::path file_path_;
};

} // namespace FileIO
} // namespace Utilities

#endif // UTILITIES_FILEIO_FILE_PATH_H