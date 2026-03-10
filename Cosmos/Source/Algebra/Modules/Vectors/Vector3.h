//------------------------------------------------------------------------------
/// \brief Implementation without inheritance, prioritizing performance through
/// simplicity.
//------------------------------------------------------------------------------
#ifndef ALGEBRA_MODULES_VECTORS_VECTOR_3_H
#define ALGEBRA_MODULES_VECTORS_VECTOR_3_H

#include <algorithm>
#include <array>
#include <cmath>
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
      elements_.fill(static_cast<Field>(0));
    }

    explicit Vector3(const std::initializer_list<Field>& entries)
    {
      std::copy(entries.begin(), entries.end(), elements_.begin());
    }

    //--------------------------------------------------------------------------
    /// \brief Copy constructor
    //--------------------------------------------------------------------------
    Vector3(const Vector3& rhs):
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

    Field x() const noexcept
    {
      return elements_[0];
    }

    Field y() const noexcept
    {
      return elements_[1];
    }

    Field z() const noexcept
    {
      return elements_[2];
    }

    bool operator==(const Vector3& rhs) const
    {
      return elements_[0] == rhs.elements_[0] &&
        elements_[1] == rhs.elements_[1] &&
        elements_[2] == rhs.elements_[2];
    }

    bool operator!=(const Vector3& rhs) const
    {
      return !(*this == rhs);
    }

    template <typename F>
    friend std::ostream& operator<<(std::ostream& os, const Vector3<F>& a);

    Vector3 operator+(const Vector3& rhs) const
    {
      return Vector3{
        elements_[0] + rhs.elements_[0],
        elements_[1] + rhs.elements_[1],
        elements_[2] + rhs.elements_[2]};
    }

    Vector3 operator-(const Vector3& rhs) const
    {
      return Vector3{
        elements_[0] - rhs.elements_[0],
        elements_[1] - rhs.elements_[1],
        elements_[2] - rhs.elements_[2]};
    }

    Vector3 operator*(const Field scalar_value) const
    {
      return Vector3{
        elements_[0] * scalar_value,
        elements_[1] * scalar_value,
        elements_[2] * scalar_value};
    }

    Vector3 operator/(const Field scalar_value) const
    {
      return Vector3{
        elements_[0] / scalar_value,
        elements_[1] / scalar_value,
        elements_[2] / scalar_value};
    }

    Vector3& operator+=(const Vector3& rhs)
    {
      elements_[0] += rhs.elements_[0];
      elements_[1] += rhs.elements_[1];
      elements_[2] += rhs.elements_[2];
      return *this;
    }

    //--------------------------------------------------------------------------
    /// A vector space equipped with a norm is a normed vector space.
    //--------------------------------------------------------------------------
    Field norm() const
    {
      return std::hypot(elements_[0], elements_[1], elements_[2]);
    }

    Field norm_squared() const
    {
      return (
        elements_[0] * elements_[0] +
        elements_[1] * elements_[1] +
        elements_[2] * elements_[2]);
    }

    //--------------------------------------------------------------------------
    /// A vector space equipped with a dot product is an inner product space.
    //--------------------------------------------------------------------------

    Field dot(const Vector3& rhs) const
    {
      return (
        elements_[0] * rhs.elements_[0] +
        elements_[1] * rhs.elements_[1] +
        elements_[2] * rhs.elements_[2]);
    }

  private:

    std::array<Field, Vector3<Field>::dimension> elements_;
};

template <typename Field>
std::ostream& operator<<(std::ostream& os, const Vector3<Field>& a)
{
  for (
    auto iter {a.elements_.begin()};
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
