#include "Utilities/DeviceManagement/GetAndSetDevice.h"
#include "gtest/gtest.h"

#include <iostream>

using Utilities::DeviceManagement::GetAndSetDevice;

namespace IntegrationTests
{
namespace Utilities
{
namespace DeviceManagement
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetAndSetDeviceTests, ConstructsWithDeviceCount)
{
  GetAndSetDevice get_and_set_device {};

  int current_device {get_and_set_device.get_current_device()};

  EXPECT_EQ(current_device, 0);

  // Because the device count can change for each system, we print out the
  // value.
  std::cout << "Device count: " << get_and_set_device.get_device_count()
            << std::endl;
}

} // namespace DeviceManagement
} // namespace Utilities
} // namespace IntegrationTests