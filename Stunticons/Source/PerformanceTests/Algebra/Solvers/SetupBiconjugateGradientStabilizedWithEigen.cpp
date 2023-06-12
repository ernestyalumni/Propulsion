#include "SetupBiconjugateGradientStabilizedWithEigen.h"
#include "Utilities/FileIO/ReadMatrixMarketFile.h"

#include <cstddef>
#include <eigen3/Eigen/Dense>
#include <stdexcept>
#include <vector>

using std::size_t;
using std::vector;

namespace PerformanceTests
{
namespace Algebra
{
namespace Solvers
{

SetupBiconjugateGradientStabilizedWithEigen::
  SetupBiconjugateGradientStabilizedWithEigen(
    const size_t rows,
    const size_t columns):
  A_{static_cast<long int>(rows), static_cast<long int>(columns)},
  solver_{},
  b_{A_.rows()},
  x_{A_.rows()}
{}

void SetupBiconjugateGradientStabilizedWithEigen::insert_into_A(
  const vector<int>& row_offsets,
  const vector<int>& column_indices,
  const vector<double>& values)
{
  for (int i {0}; i < row_offsets.size() - 1; ++i)
  {
    for (int j {row_offsets.at(i)}; j < row_offsets.at(i + 1); ++j)
    {
      A_.insert(i, column_indices.at(j)) = values.at(j);
    }
  }

  // Compress the matrix to optimize the storage.
  A_.makeCompressed();
}

void SetupBiconjugateGradientStabilizedWithEigen::insert_into_b(
  const vector<double>& b_values)
{
  for (size_t i {0}; i < b_values.size(); ++i)
  {
    b_[i] = b_values.at(i);
  }
}

void SetupBiconjugateGradientStabilizedWithEigen::setup_solving()
{
  solver_.compute(A_);

  if (solver_.info() != Eigen::Success)
  {
    throw std::runtime_error("Decomposition failed\n");
  }
}

void SetupBiconjugateGradientStabilizedWithEigen::solve()
{
  x_ = solver_.solve(b_);

  if (solver_.info() != Eigen::Success)
  {
    throw std::runtime_error("Solving failed\n");
  }
}

} // namespace Solvers
} // namespace Algebra
} // namespace PerformanceTests