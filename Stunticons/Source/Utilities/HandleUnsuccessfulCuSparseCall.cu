#include "HandleUnsuccessfulCuSparseCall.h"

#include <cusparse.h>
#include <iostream> // std::cerr
#include <string>

using std::cerr;

namespace Utilities
{

HandleUnsuccessfulCuSparseCall::HandleUnsuccessfulCuSparseCall(
  const std::string& error_message
  ):
  error_message_{error_message},
  cusparse_status_{CUSPARSE_STATUS_SUCCESS}
{}

void HandleUnsuccessfulCuSparseCall::operator()(
  const cusparseStatus_t cusparse_status)
{
  cusparse_status_ = cusparse_status;

  if (!is_cusparse_success())
  {
    cerr << error_message_ << " (error code " <<
      cusparseGetErrorString(cusparse_status_) << ")!\n";
  }
}

} // namespace Utilities