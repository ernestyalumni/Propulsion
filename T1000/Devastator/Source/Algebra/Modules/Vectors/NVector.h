//------------------------------------------------------------------------------
/// \brief Implementation without inheritance, prioritizing performance through
/// simplicity.
/// \details Consider using std::valarray instead.
//------------------------------------------------------------------------------
#ifndef ALGEBRA_MODULES_VECTORS_N_VECTOR_H
#define ALGEBRA_MODULES_VECTORS_N_VECTOR_H

#include <algorithm>
#include <cstddef>
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

    template <std::size_t M, typename F>
    friend NVector<M, F> operator+(
      const NVector<M, F>& x1,
      const NVector<M, F>& x2);

    template <std::size_t M, typename F>
    friend NVector<M, F> operator-(
      const NVector<M, F>& x1,
      const NVector<M, F>& x2);

  private:

    std::vector<Field> components_;
};

//------------------------------------------------------------------------------
/// \details Binary Operators, Prefer implementing binary operators as free
/// functions.
/// \ref Ch. 2, Discovering Modern C++, @nd. Ed. Peter Gottschling.
//------------------------------------------------------------------------------

template <std::size_t N, typename Field = double>
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

template <std::size_t N, typename Field = double>
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

} // namespace Vectors
} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_VECTORS_N_VECTOR_H
