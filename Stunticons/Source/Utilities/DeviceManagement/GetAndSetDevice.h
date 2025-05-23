#ifndef UTILITIES_DEVICE_MANAGEMENT_GET_AND_SET_DEVICE_H
#define UTILITIES_DEVICE_MANAGEMENT_GET_AND_SET_DEVICE_H

namespace Utilities
{
namespace DeviceManagement
{

class GetAndSetDevice
{
  public:

    GetAndSetDevice();

    inline int get_device_count() const noexcept
    {
      return device_count_;
    }

    void set_device(const int device_index) const;

    int get_current_device();

  private:

    int device_count_;
    int current_device_;
};

} // namespace DeviceManagement
} // namespace Utilities

#endif // UTILITIES_DEVICE_MANAGEMENT_GET_AND_SET_DEVICE_H