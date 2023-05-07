#include "GetCUDADeviceProperties.h"

#include <cuda_runtime.h>
#include <iostream> // std::cerr

using std::cerr;
using std::cout;

namespace Utilities
{
namespace DeviceManagement
{

GetCUDADeviceProperties::GetCUDADeviceProperties():
  device_count_{-1},
  cuda_device_properties_{},
  abridged_properties_{}
{
  // Returns number of compute-capable devices.
  const cudaError_t device_count_error {cudaGetDeviceCount(&device_count_)};

  if (device_count_error != cudaSuccess)
  {
    if (device_count_error == cudaErrorNoDevice)
    {
      cerr << "CUDA error for No Device (error code " <<
        cudaGetErrorString(device_count_error) << ")!\n";
    }
    else
    {
      cerr << "Failed to get CUDA device count (error code " <<
        cudaGetErrorString(device_count_error) << ")!\n";      
    }
  }

  for (int i {0}; i < device_count_; ++i)
  {
    cudaDeviceProp device_properties {};
    const cudaError_t properties_error {
      cudaGetDeviceProperties(&device_properties, i)};

    if (properties_error != cudaSuccess)
    {
      cerr << "Failed to get CUDA device properties (error code " <<
        cudaGetErrorString(properties_error) << ")!\n";
    }
    else
    {
      cuda_device_properties_.emplace_back(device_properties);
      abridged_properties_.emplace_back(
        AbridgedProperties{
          device_properties.major,
          device_properties.minor,
          device_properties.multiProcessorCount,
          device_properties.sharedMemPerMultiprocessor,
          device_properties.totalGlobalMem,
          device_properties.warpSize});
    }
  }
}

void GetCUDADeviceProperties::pretty_print_abridged_properties()
{
  for (int i {0}; i < device_count_; ++i)
  {
    cout << "Device " << i << ": " << cuda_device_properties_.at(i).name << "\n";
    cout << "Compute capability: " << cuda_device_properties_.at(i).major << "." <<
      cuda_device_properties_.at(i).minor << "\n";
    cout << "Multiprocessor count: " <<
      abridged_properties_.at(i).multi_processor_count_ << "\n";
    cout << "Shared memory per multiprocessor in bytes: " <<
      abridged_properties_.at(i).shared_memory_bytes_ << "\n";
    cout << "Total global memory: " <<
      abridged_properties_.at(i).total_global_memory_bytes_ << "\n";
  }
}

} // namespace DeviceManagement
} // namespace Utilities
