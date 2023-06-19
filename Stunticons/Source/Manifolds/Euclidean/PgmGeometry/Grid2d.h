#ifndef MANIFOLDS_EUCLIDEAN_PGM_GEOMETRY_GRID2D_H
#define MANIFOLDS_EUCLIDEAN_PGM_GEOMETRY_GRID2D_H

#include <cstddef>
#include <vector>

namespace Manifolds
{
namespace Euclidean
{
// Pgm refers to the .pgm suffix used for the "configuration" files here,
// https://github.com/yuphin/turbulent-cfd/tree/master/example_cases
// within each example case.
namespace PgmGeometry
{

//------------------------------------------------------------------------------
/// \details i = 0, 1, ... M + 1, so |I| = M + 2
/// j = 0, 1, ... N + 1, so |J| = N + 2
//------------------------------------------------------------------------------
class Grid2d
{
  public:

    Grid2d() = delete;

    Grid2d(const std::size_t M, const std::size_t N);

    ~Grid2d() = default;

    inline int get(const std::size_t i, const std::size_t j) const
    {
      return values_[i + j * (M_ + 2)];
    }

    inline int& get(const std::size_t i, const std::size_t j)
    {
      return values_[i + j * (M_ + 2)];
    }

    inline int at(const std::size_t i, const std::size_t j) const
    {
      return values_.at(i + j * (M_ + 2));
    }

    inline int& at(const std::size_t i, const std::size_t j)
    {
      return values_.at(i + j * (M_ + 2));
    }

    std::size_t get_M() const
    {
      return M_;
    }

    std::size_t get_N() const
    {
      return N_;
    }

  private:

    std::vector<int> values_;

    std::size_t M_;    
    std::size_t N_;
};

} // namespace PgmGeometry
} // namespace Euclidean
} // namespace Manifolds

#endif // MANIFOLDS_EUCLIDEAN_PGM_GEOMETRY_GRID2D_H