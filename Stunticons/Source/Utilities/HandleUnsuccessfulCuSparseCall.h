#ifndef UTILITIES_HANDLE_UNSUCCESSFUL_CUSPARSE_CALL_H
#define UTILITIES_HANDLE_UNSUCCESSFUL_CUSPARSE_CALL_H

#include <cusparse.h>
#include <string>

namespace Utilities
{

class HandleUnsuccessfulCuSparseCall
{
  public:

    inline static const std::string default_error_message_ {
      "cuSPARSE success status was not returned."};

    HandleUnsuccessfulCuSparseCall(
      const std::string& error_message = default_error_message_);

    ~HandleUnsuccessfulCuSparseCall() = default;

    inline bool is_cusparse_success()
    {
      return cusparse_status_ == CUSPARSE_STATUS_SUCCESS;
    }

    void operator()(const cusparseStatus_t cusparse_status);

    cusparseStatus_t get_cusparse_status() const
    {
      return cusparse_status_;
    }

  private:

    std::string error_message_;

    cusparseStatus_t cusparse_status_;
};

} // namespace Utilities

#endif // UTILITIES_HANDLE_UNSUCCESSFUL_CUSPARSE_CALL_H
