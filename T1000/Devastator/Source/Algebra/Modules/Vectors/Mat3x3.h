//------------------------------------------------------------------------------
/// \brief Implementation without inheritance, prioritizing performance through
/// simplicity.
/// \ref 17.5.1 Copy Ch. 17 Construction, Cleanup, Copy, and Move; 
///   Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup
//------------------------------------------------------------------------------

#ifndef ALGEBRA_MODULES_VECTORS_MAT_3X3_H
#define ALGEBRA_MODULES_VECTORS_MAT_3X3_H

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

template <typename Field>
class Mat3x3
{
  public:

    static constexpr std::size_t number_of_elements {9};
    static constexpr std::size_t dimension {3};

    Mat3x3() = default;

    explicit Mat3x3(std::initializer_list<Field> entries)
    {
      std::copy(entries.begin(), entries.end(), entries_.begin());
    }

    virtual Mat3x3() = default;

    //--------------------------------------------------------------------------
    /// \details row-major order, 0-indexed
    /// Does no bounds checking.
    //--------------------------------------------------------------------------
    T get_entry(const std::size_t i, const std::size_t j) const noexcept
    {
      return entries_[Mat3x3<Field>::dimension * i + j];
    }


    template <typename F>
    friend std::ostream& operator<<(std::ostream& os, const Mat3x3<F>& A);

  private:

    std::array<Field, Mat3x3<Field>::number_of_elements> entries_;
};

template <typename Field>
std::ostream& operator<<(std::ostream& os, const Mat3x3<Field>& A)
{
  for (std::size_t i {0}; i < Mat3x3<Field>::dimension; ++i)
  {
    for (std::size_t j {0}; j < Mat3x3<Field>::dimension; ++j)
    {
      os << A.get_entry(i, j) << ' ';
    }
    os << '\n';
  }
  os << '\n';

  return os;
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_VECTORS_MAT_3X3_H
