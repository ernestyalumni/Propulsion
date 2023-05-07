#include "GetCUDADeviceProperties.h"

#include <iostream>

using Utilities::DeviceManagement::GetCUDADeviceProperties;

int main()
{
  GetCUDADeviceProperties device_properties {};

  std::cout << "Number of Devices " << device_properties.get_device_count() <<
    "\n";

  if (device_properties.get_device_count() > 0)
  {
    device_properties.pretty_print_abridged_properties();
  }
}