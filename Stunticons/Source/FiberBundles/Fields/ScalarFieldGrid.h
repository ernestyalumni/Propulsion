#ifndef FIBER_BUNDLES_FIELDS_SCALAR_FIELD_GRID_H
#define FIBER_BUNDLES_FIELDS_SCALAR_FIELD_GRID_H

#include <cstddef>
#include <vector>

namespace FiberBundles
{
namespace Fields
{

//------------------------------------------------------------------------------
/// \details A scalar field "lives above" a smooth manifold. In this case, the
/// manifold is discretized and is a 2-dim. grid.
/// i = 0, 1, ... M -1, so |I| = M
/// j = 0, 1, ... N -1, so |J| = N
//------------------------------------------------------------------------------
class ScalarFieldGrid
{
  public:

    ScalarFieldGrid() = delete;

    //--------------------------------------------------------------------------
    /// \details This creates an array of M * N elements, which can
    /// be accessed in a "2-dimensional grid way."
    /// Indicies ranges are as follows: i = 0, 1, ... M - 1, j = 0, 1, ... N - 1
    //--------------------------------------------------------------------------
    ScalarFieldGrid(
      const std::size_t M,
      const std::size_t N,
      const double initial_value = 0.0);

    ~ScalarFieldGrid() = default;

    inline double get(const std::size_t i, const std::size_t j) const
    {
      return values_[i + j * M_];
    }

    inline double& get(const std::size_t i, const std::size_t j)
    {
      return values_[i + j * M_];
    }

    inline double at(const std::size_t i, const std::size_t j) const
    {
      return values_.at(i + j * M_);
    }

    inline double& at(const std::size_t i, const std::size_t j)
    {
      return values_.at(i + j * M_);
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

    std::vector<double> values_;

    std::size_t M_;    
    std::size_t N_;
};

} // namespace Fields
} // namespace FiberBundles

#endif // FIBER_BUNDLES_FIELDS_SCALAR_FIELD_GRID_H
