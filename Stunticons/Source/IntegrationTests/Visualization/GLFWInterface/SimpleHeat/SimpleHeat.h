#ifndef INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_SIMPLE_HEAT_SIMPLE_HEAT_H
#define INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_SIMPLE_HEAT_SIMPLE_HEAT_H

#include "DataStructures/Array.h"
#include "IntegrationTests/Visualization/GLFWInterface/SimpleHeat/Benchmarking.h"

#include <cstddef>

namespace IntegrationTests
{
namespace Visualization
{
namespace GLFWInterface
{
namespace SimpleHeat
{

class SimpleHeat
{
  public:

    template <typename T>
    using Array = DataStructures::Array<T>;

    inline static constexpr std::size_t dimension_ {1024};
    inline static constexpr std::size_t threads_ {16};
    inline static constexpr float speed_ {0.25f};
    inline static constexpr float max_temperature_ {1.0f};
    inline static constexpr float min_temperature_ {0.0001f};

    inline static constexpr std::size_t time_steps_per_frame_ {90};

    inline static constexpr std::size_t get_image_size()
    {
      // Assume float == 4 chars in size (i.e., RGBA).
      return dimension_ * dimension_ * 4;
    }

    SimpleHeat();
    ~SimpleHeat() = default;

    //--------------------------------------------------------------------------
    /// chosen 90 time steps per frame; manually ("experimentally") determined
    /// as reasonable tradeoff between having to download a bitmap image for
    /// every time step and computing too many time steps per frame, resulting
    /// in a jerky animation you could change this, if more concerned with
    /// getting output of each simulation step than animating results in real
    /// time, e.g. only single step on each frame
    /// See pp. 123 of Sanders and Kandrot
    //--------------------------------------------------------------------------
    void run(uchar4* ptr, int ticks);

    Array<unsigned char> output_graphics_;
    Array<float> input_;
    Array<float> output_;
    // For initial value conditions.
    Array<float> constant_input_;

  protected:

    bool create_initial_conditions();

    Benchmarking benchmarks_;
};

} // namespace SimpleHeat
} // namespace GLFWInterface
} // namespace Visualization
} // namespace IntegrationTests

#endif // INTEGRATION_TESTS_VISUALIZATION_GLFW_INTERFACE_SIMPLE_HEAT_SIMPLE_HEAT_H