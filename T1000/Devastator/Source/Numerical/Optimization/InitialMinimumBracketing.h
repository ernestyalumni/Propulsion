#ifndef NUMERICAL_OPTIMIZATION_INITIAL_MINIMUM_BRACKETING_H
#define NUMERICAL_OPTIMIZATION_INITIAL_MINIMUM_BRACKETING_H

#include "FindParabolicInterpolation.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility> // std::swap

namespace Numerical
{
namespace Optimization
{

struct InitialMinimumBracketing
{
  template <
    typename F,
    typename T = double,
    typename = typename std::enable_if_t<std::is_floating_point<T>::value>
    >
  static std::optional<std::array<T, 3>> is_minimum_found(
    const T a,
    const T b,
    const T c,
    const T fa,
    const T fb,
    const T fc,
    const T u,
    const T u_limit,
    F& input_function)
  {
    assert(((fa >= fb) && (fb >= fc)) && ((a != b) && (b != c)));

    // Check if the parabolic fit exceeded its user-defined limit.
    if ((c - u_limit) * (u_limit - u) > 0.0)
    {
      return std::nullopt;
    }

    const T fu {input_function(u)};

    if (((a - u) * (u - b) > static_cast<T>(0)) && fu < fb)
    {
      return std::make_optional<std::array<T, 3>>({a, u, b});
    }

    if (((b - u) * (u - c) > static_cast<T>(0)) &&
      ((fb < fu) || (fu < fc)))
    {
      return std::make_optional<std::array<T, 3>>({b, u, c});
    }

    if (((b - c) * (c - u) > static_cast<T>(0)) &&
      ((c - u) * (u - u_limit) > static_cast<T>(0)) &&
      (fu > fc))
    {
      return std::make_optional<std::array<T, 3>>({b, c, u});
    }

    return std::nullopt;
  }

  template <
    typename F,
    typename T = double,
    typename = typename std::enable_if_t<std::is_floating_point<T>::value>
    >
  static std::optional<std::array<T, 3>> bracket_minimum(
    const T a,
    const T b,
    F& input_function,
    std::size_t number_of_tries = 10000
    )
  {
    static constexpr double GOLD {1.618034};
    static constexpr double GLIMIT {100.0};

    T ax {a};
    T bx {b};

    T fa {input_function(ax)};
    T fb {input_function(bx)};

    if (fa < fb)
    {
      std::swap(ax, bx);
      std::swap(fa, fb);
    }

    // First guess for c.
    const T cx {bx + GOLD * (bx - ax)};
    const T fc {input_function(cx)};

    std::array<T, 3> store_coordinates {ax, bx, cx};
    std::array<T, 3> store_values {fa, fb, fc};

    for (std::size_t i {0}; i < number_of_tries; ++i)
    {
      if (store_values[1] < store_values[2])
      {
        return std::make_optional<std::array<T, 3>>(store_coordinates);
      }

      const auto parabolic_results = FindParabolicInterpolation::find_extrema(
        store_coordinates[0],
        store_coordinates[1],
        store_coordinates[2],
        store_values[0],
        store_values[1],
        store_values[2]);

      const T u_limit {
        store_coordinates[1] +
          GLIMIT * (store_coordinates[2] - store_coordinates[1])};

      const auto minimum_result = is_minimum_found(
        store_coordinates[0],
        store_coordinates[1],
        store_coordinates[2],
        store_values[0],
        store_values[1],
        store_values[2],
        parabolic_results.extrema_position_,
        u_limit,
        input_function);

      if (minimum_result.has_value())
      {
        return minimum_result;
      }

      assert(!minimum_result.has_value());

      // If parabolic fit exceeds our user-defined limit:
      if ((store_coordinates[2] - u_limit) *
        (u_limit - parabolic_results.extrema_position_) > 0.0)
      {
        shift_elements_left(store_coordinates, u_limit);
        shift_elements_left(store_values, input_function(u_limit));
      }
      // TODO: Consider deleting this because condition because it's necessary
      // to jump to where the parabolic fit is.
      //else if ((store_coordinates[2] - parabolic_results.extrema_position_) *
      //  (parabolic_results.extrema_position_ - u_limit) > 0.0 &&
      //    (input_function(parabolic_results.extrema_position_) <
      //      store_values[2]))
      //{
      //  const T new_u {parabolic_results.extrema_position_ + GOLD * (
      //    parabolic_results.extrema_position_ - store_values[2])};

      //  shift_elements_left(store_coordinates, new_u);
      //  shift_elements_left(store_values, input_function(new_u));
      //}
      else
      {
        const T cx {
          store_coordinates[2] +
            GOLD * (store_coordinates[2] - store_coordinates[1])};

        if (cx >= std::numeric_limits<T>::max() ||
          cx <= std::numeric_limits<T>::lowest())
        {
          return std::nullopt;
        }

        shift_elements_left(store_coordinates, cx);
        shift_elements_left(store_values, input_function(cx));
      }
    }

    return std::nullopt;
  }

  template <
    typename T = double,
    typename = typename std::enable_if_t<std::is_floating_point<T>::value>
    >
  static void shift_elements_left(std::array<T, 3>& input, const T new_value)
  {
    input[0] = input[1];
    input[1] = input[2];
    input[2] = new_value;
  }
};

} // namespace Optimization
} // namespace Numerical

#endif // NUMERICAL_OPTIMIZATION_INITIAL_MINIMUM_BRACKETING_H

