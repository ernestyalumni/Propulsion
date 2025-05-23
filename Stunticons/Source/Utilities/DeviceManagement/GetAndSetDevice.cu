#include "GetAndSetDevice.h"
#include "Utilities/HandleUnsuccessfulCudaCall.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

using Utilities::HandleUnsuccessfulCudaCall;

namespace Utilities
{
namespace DeviceManagement
{

GetAndSetDevice::GetAndSetDevice():
  device_count_{-1}
{
  HandleUnsuccessfulCUDACall handle_get_device_count {
    "Failed to get CUDA device count"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_get_device_count,
    cudaGetDeviceCount(&device_count_));

  get_current_device();
}

int GetAndSetDevice::get_current_device()
{
  HandleUnsuccessfulCUDACall handle_get_current_device {
    "Failed to get current CUDA device"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_get_current_device,
    cudaGetDevice(&current_device_));

  return current_device_;
}

void GetAndSetDevice::set_device(const int device_index) const
{
  if (device_index < 0 || device_index >= device_count_)
  {
    throw std::invalid_argument(
      "Invalid device index" + std::to_string(device_index) +
        " for device count" + std::to_string(device_count_));
  }

  HandleUnsuccessfulCUDACall handle_set_device {
    "Failed to set CUDA device"};

  HANDLE_UNSUCCESSFUL_CUDA_CALL_WITH_LOCATION(
    handle_set_device,
    cudaSetDevice(device_index));
}

} // namespace DeviceManagement
} // namespace Utilities
