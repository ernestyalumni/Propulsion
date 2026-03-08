//------------------------------------------------------------------------------
/// \brief Implementation without inheritance, prioritizing performance through
/// simplicity.
/// \details Consider using std::valarray. But std::vector could be competitive
/// with std::valarray.
/// TODO: Measure performance.
/// \ref https://stackoverflow.com/questions/1602451/c-valarray-vs-vector
//------------------------------------------------------------------------------
#ifndef ALGEBRA_MODULES_VECTORS_N_VECTOR_H
#define ALGEBRA_MODULES_VECTORS_N_VECTOR_H

#include <algorithm>
#include <cstddef>
#include <functional> // std::placeholders
#include <initializer_list>
#include <iostream>
#include <vector>

namespace Algebra
{
namespace Modules
{
namespace Vectors
{

template <std::size_t N, typename Field = double>
class NVector
{
  public:

    static constexpr std::size_t dimension {N};

    NVector():
      components_(N, static_cast<Field>(0))
    {}

    NVector(const std::initializer_list<Field>& input):
      components_{input}
    {
      components_.reserve(N);
    }

    explicit NVector(const Field initial_value):
      components_(N, initial_value)
    {}

    // Copy ctor.
    NVector(const NVector& other) = default;

    // Copy assignment.
    NVector& operator=(const NVector& other) = default;

    // Move ctor.
    NVector(NVector&& other) = default;

    // Move assignment.
    NVector& operator=(NVector&& other) = default;

    virtual ~NVector() = default;

    Field& operator[](const std::size_t index)
    {
      return components_[index];
    }

    const Field& operator[](const std::size_t index) const
    {
      return components_[index];
    }

    NVector<N, Field>& operator+=(const NVector<N, Field>& x)
    {
      std::transform(
        components_.begin(),
        components_.end(),
        x.components_.begin(),
        components_.begin(),
        std::plus<Field>());

      return *this;
    }

    NVector<N, Field>& operator-=(const NVector<N, Field>& x)
    {
      std::transform(
        components_.begin(),
        components_.end(),
        x.components_.begin(),
        components_.begin(),
        std::minus<Field>());

      return *this;
    }

    NVector<N, Field>& operator*=(const Field h)
    {
      std::transform(
        components_.begin(),
        components_.end(),
        components_.begin(),
        [h](const Field element)
        {
          return element * h;
        });

      return *this;
    }

    template <std::size_t M, typename F>
    friend NVector<M, F> operator+(
      const NVector<M, F>& x1,
      const NVector<M, F>& x2);

    template <std::size_t M, typename F>
    friend NVector<M, F> operator+(const F value, const NVector<M, F>& x);

    template <std::size_t M, typename F>
    friend NVector<M, F> operator+(const NVector<M, F>& x, const F value);

    template <std::size_t M, typename F>
    friend NVector<M, F> operator-(
      const NVector<M, F>& x1,
      const NVector<M, F>& x2);

    template <std::size_t M, typename F>
    friend NVector<M, F> operator-(const NVector<M, F>& x, const F value);

    //--------------------------------------------------------------------------
    /// \brief Left scalar multiplication.
    //--------------------------------------------------------------------------
    template <std::size_t M, typename F>
    friend NVector<M, F> operator*(
      const F scalar_value,
      const NVector<M, F>& x);

    //--------------------------------------------------------------------------
    /// \brief Right scalar division.
    //--------------------------------------------------------------------------
    template <std::size_t M, typename F>
    friend NVector<M, F> operator/(
      const NVector<M, F>& x,
      const F scalar_value);

  private:

    std::vector<Field> components_;
};

//------------------------------------------------------------------------------
/// \details Binary Operators, Prefer implementing binary operators as free
/// functions.
/// \ref Ch. 2, Discovering Modern C++, @nd. Ed. Peter Gottschling.
//------------------------------------------------------------------------------

template <std::size_t N, typename Field>
inline NVector<N, Field> operator+(
  const NVector<N, Field>& x1,
  const NVector<N, Field>& x2)
{
  NVector<N, Field> result {};
  std::transform(
    x1.components_.begin(),
    x1.components_.end(),
    x2.components_.begin(),
    result.components_.begin(),
    std::plus<Field>());

  return result;
}

template <std::size_t N, typename Field>
inline NVector<N, Field> operator+(
  const Field value,
  const NVector<N, Field>& x)
{
  NVector<N, Field> result {};
  std::transform(
    x.components_.begin(),
    x.components_.end(),
    result.components_.begin(),
    std::bind(std::plus<Field>(), std::placeholders::_1, value));
  return result;
}

template <std::size_t N, typename Field>
inline NVector<N, Field> operator+(
  const NVector<N, Field>& x,
  const Field value)
{
  return value + x;
}

template <std::size_t N, typename Field>
inline NVector<N, Field> operator-(
  const NVector<N, Field>& x,
  const Field value)
{
  NVector<N, Field> result {};
  std::transform(
    x.components_.begin(),
    x.components_.end(),
    result.components_.begin(),
    std::bind(std::minus<Field>(), std::placeholders::_1, value));
  return result;
}

template <std::size_t N, typename Field>
inline NVector<N, Field> operator-(
  const NVector<N, Field>& x1,
  const NVector<N, Field>& x2)
{
  NVector<N, Field> result {};
  std::transform(
    x1.components_.begin(),
    x1.components_.end(),
    x2.components_.begin(),
    result.components_.begin(),
    std::minus<Field>());

  return result;
}

template <std::size_t N, typename Field>
inline NVector<N, Field> operator*(
  const Field scalar_value,
  const NVector<N, Field>& x)
{
  NVector<N, Field> result {};
  std::transform(
    x.components_.begin(),
    x.components_.end(),
    result.components_.begin(),
    std::bind(std::multiplies<Field>(), std::placeholders::_1, scalar_value));
  return result;
}

template <std::size_t N, typename Field>
inline NVector<N, Field> operator/(
  const NVector<N, Field>& x,
  const Field scalar_value)
{
  NVector<N, Field> result {};
  std::transform(
    x.components_.begin(),
    x.components_.end(),
    result.components_.begin(),
    std::bind(std::divides<Field>(), std::placeholders::_1, scalar_value));
  return result;
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_VECTORS_N_VECTOR_H
