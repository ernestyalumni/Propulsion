//------------------------------------------------------------------------------
/// \brief Implementation without inheritance, prioritizing performance through
/// simplicity.
//------------------------------------------------------------------------------
#ifndef ALGEBRA_MODULES_VECTORS_VECTOR_3_H
#define ALGEBRA_MODULES_VECTORS_VECTOR_3_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <iostream>

namespace Algebra
{
namespace Modules
{
namespace Vectors
{

template <typename Field = double>
class Vector3
{
  public:

    static constexpr std::size_t dimension {3};

    Vector3():
      elements_{}
    {
      elements.fill(static_cast<Field>(0));
    }

    explicit Vector3(const std::initializer_list<Field>& entries)
    {
      std::copy(entries.begin(), entries.end(), elements_.begin());
    }

    //--------------------------------------------------------------------------
    /// \brief Copy constructor
    //--------------------------------------------------------------------------
    Vector3(const Vector& rhs):
      elements_{rhs.elements_}
    {}

    //--------------------------------------------------------------------------
    /// \brief Copy assignment
    //--------------------------------------------------------------------------
    Vector3& operator=(const Vector3& rhs)
    {
      elements_ = rhs.elements_;

      return *this;
    }

    //--------------------------------------------------------------------------
    /// \brief Move Constructor
    /// \ref https://stackoverflow.com/questions/22613991/move-constructors-and-stdarray
    /// https://isocpp.org/blog/2014/03/quick-q-is-stdarrayt-movable-any-better-than-a-plan-c-array-stackoverflow
    //--------------------------------------------------------------------------
    Vector3(Vector3&& rhs):
      elements_{std::move(rhs.elements_)}
    {}

    //--------------------------------------------------------------------------
    /// \brief Move Assignment
    //--------------------------------------------------------------------------
    Vector3& operator=(Vector3&& rhs)
    {
      // Exchanges contents of the container with those other rhs.elements_.
      // Doesn't cause iterators and references to associate with the other
      // container.
      elements_.swap(rhs.elements_);
      
      return *this;
    }

    virtual ~Vector3() = default;

    //--------------------------------------------------------------------------
    /// \details row_major order, 0-indexed.
    /// Does no bounds checking. 
    //--------------------------------------------------------------------------
    Field get_entry(const std::size_t i) const noexcept
    {
      return elements_[i];
    }

    template <typename F>
    friend std::ostream& operator<<(std::ostream& os, const Vector3<F>& a);

  private:

    std::array<Field, Vector3<Field>::dimension> elements_;
};

template <typename Field>
std::ostream& operator<<(std::ostream& os, const Vector3<Field>& a)
{
  for (
    const auto iter {a.elements_.begin()};
    iter != a.elements_.end();
    ++iter)
  {
    os << *iter << ' ';
  }

  os << '\n';

  return os;
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_VECTORS_VECTOR_3_H
