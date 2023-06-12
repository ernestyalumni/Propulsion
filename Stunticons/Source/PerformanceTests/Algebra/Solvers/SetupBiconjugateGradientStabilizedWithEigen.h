#ifndef PERFORMANCE_TESTS_ALGEBRA_SOLVERS_SETUP_BICONJUGATE_GRADIENT_STABILIZED_WITH_EIGEN_H
#define PERFORMANCE_TESTS_ALGEBRA_SOLVERS_SETUP_BICONJUGATE_GRADIENT_STABILIZED_WITH_EIGEN_H

#include "Utilities/FileIO/ReadMatrixMarketFile.h"

#include <cstddef>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/IterativeLinearSolvers>
#include <vector>

namespace PerformanceTests
{
namespace Algebra
{
namespace Solvers
{

class SetupBiconjugateGradientStabilizedWithEigen
{
  public:

    SetupBiconjugateGradientStabilizedWithEigen() = delete;

    SetupBiconjugateGradientStabilizedWithEigen(
      const std::size_t rows,
      const std::size_t columns);

    ~SetupBiconjugateGradientStabilizedWithEigen() = default;

    void insert_into_A(
      const std::vector<int>& row_offsets,
      const std::vector<int>& column_indices,
      const std::vector<double>& values);

    void insert_into_b(const std::vector<double>& b_values);

    void setup_solving();

    void solve();

    Eigen::SparseMatrix<double, Eigen::RowMajor> A_;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver_;
    Eigen::VectorXd b_;
    Eigen::VectorXd x_;
};

} // namespace Solvers
} // namespace Algebra
} // namespace PerformanceTests

#endif // PERFORMANCE_TESTS_ALGEBRA_SOLVERS_SETUP_BICONJUGATE_GRADIENT_STABILIZED_WITH_EIGEN_H