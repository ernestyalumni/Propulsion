#include "Algebra/Modules/Matrices/CompressedSparseRow.h"
#include "Algebra/Modules/Matrices/HostCompressedSparseRow.h"
#include "Algebra/Modules/Morphisms/SparseMatrixMorphism.h"
#include "Algebra/Modules/Vectors/CuBLASVectorOperations.h"
#include "Algebra/Solvers/BiconjugateGradientStabilized.h"
#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <tuple> // std::get
#include <vector>

using Array = Algebra::Modules::Vectors::DoubleArray;
using BiconjugateGradientStabilized =
  Algebra::Solvers::BiconjugateGradientStabilized;
using CompressedSparseRowMatrix =
	Algebra::Modules::Matrices::SparseMatrices::DoubleCompressedSparseRowMatrix;
using CuBLASVectorOperations =
	Algebra::Modules::Vectors::DoubleCuBLASVectorOperations;
using DenseVector =
	Algebra::Modules::Matrices::SparseMatrices::DoubleDenseVector;
using HostCompressedSparseRowMatrix =
	Algebra::Modules::Matrices::SparseMatrices::
    DoubleHostCompressedSparseRowMatrix;
using SparseMatrixMorphismOnDenseVector =
	Algebra::Modules::Morphisms::DoubleSparseMatrixMorphismOnDenseVector;

using std::get;
using std::size_t;
using std::vector;

namespace GoogleUnitTests
{
namespace Algebra
{
namespace Solvers
{

struct SetupBiconjugateGradientStabilizedTest
{
  size_t M_;
  size_t number_of_nonzero_elements_;

  HostCompressedSparseRowMatrix h_csr_;
  CompressedSparseRowMatrix A_;
  DenseVector b_;
  DenseVector x_;
  DenseVector Ax_;
  SparseMatrixMorphismOnDenseVector morphism_;
  Array r_;
  CuBLASVectorOperations operations_;
  DenseVector p_;
  DenseVector s_;

  SetupBiconjugateGradientStabilizedTest(
    const size_t M,
    const size_t number_of_nonzero_elements,
    const double alpha=1.0,
    const double beta=0.0
    ):
    M_{M},
    number_of_nonzero_elements_{number_of_nonzero_elements},
    h_csr_{M_, M_, number_of_nonzero_elements},
    A_{M_, M_, number_of_nonzero_elements},
    b_{M_},
    x_{M_},
    Ax_{M_},
    morphism_{alpha, beta},
    r_{M_},
    operations_{},
    p_{M_},
    s_{M_}
  {}

  ~SetupBiconjugateGradientStabilizedTest() = default;
};

struct SetupBiconjugateGradientStabilizedExample :
  SetupBiconjugateGradientStabilizedTest
{
  SetupBiconjugateGradientStabilizedExample():
    SetupBiconjugateGradientStabilizedTest{2, 4}
  {
    h_csr_.copy_values(vector<double>{5, 1, 1, 8});
    h_csr_.copy_row_offsets(vector<int>{0, 2, 4});
    h_csr_.copy_column_indices(vector<int>{0, 1, 0, 1});

    A_.copy_host_input_to_device(h_csr_);

    morphism_.buffer_size(A_, x_, Ax_);

    vector<double> h_b {3.0, 2.0};

    b_.copy_host_input_to_device(h_b);    
  }

  ~SetupBiconjugateGradientStabilizedExample() = default;    
};

struct SetupBiconjugateGradientStabilizedExample1 :
  SetupBiconjugateGradientStabilizedTest
{
  SetupBiconjugateGradientStabilizedExample1():
    SetupBiconjugateGradientStabilizedTest{4, 12}
  {
    h_csr_.copy_values(
      vector<double>{4, -1, -1, -1, 4, -1, -1, 4, -1, -1, -1, 4});
    h_csr_.copy_row_offsets(vector<int>{0, 3, 6, 9, 12});
    h_csr_.copy_column_indices(vector<int>{0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3});

    A_.copy_host_input_to_device(h_csr_);

    morphism_.buffer_size(A_, x_, Ax_);

    vector<double> h_b {0, 6, 0, 6};

    b_.copy_host_input_to_device(h_b);    
  }

  ~SetupBiconjugateGradientStabilizedExample1() = default;    
};

struct SetupBiconjugateGradientStabilizedExample2 :
  SetupBiconjugateGradientStabilizedTest
{
  SetupBiconjugateGradientStabilizedExample2():
    SetupBiconjugateGradientStabilizedTest{3, 7}
  {
    h_csr_.copy_values(
      vector<double>{2, -1, -1, 2, -1, -1, 2});
    h_csr_.copy_row_offsets(vector<int>{0, 2, 5, 7});
    h_csr_.copy_column_indices(vector<int>{0, 1, 0, 1, 2, 1, 2});

    A_.copy_host_input_to_device(h_csr_);

    morphism_.buffer_size(A_, x_, Ax_);

    vector<double> h_b {1, 0, -1};

    b_.copy_host_input_to_device(h_b);    
  }

  ~SetupBiconjugateGradientStabilizedExample2() = default;    
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BiconjugateGradientStabilizedTests, Constructible)
{
  SetupBiconjugateGradientStabilizedExample setup {};

  BiconjugateGradientStabilized cg {
    setup.A_,
    setup.b_,
    setup.morphism_,
    setup.operations_};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BiconjugateGradientStabilizedTests, SolveWorksOnExample)
{
  SetupBiconjugateGradientStabilizedExample setup {};

  BiconjugateGradientStabilized cg {
    setup.A_,
    setup.b_,
    setup.morphism_,
    setup.operations_};

  cg.create_default_initial_guess(setup.x_);

  const auto result = cg.solve(
    setup.x_,
    setup.Ax_,
    setup.r_,
    setup.p_,
    setup.s_);

  EXPECT_TRUE(get<0>(result));
  EXPECT_EQ(get<1>(result), 1);

  vector<double> h_x_output (setup.M_, 0.0);
  vector<double> y (setup.M_, 0.0);

  setup.x_.copy_device_output_to_host(h_x_output);

  setup.h_csr_.multiply(h_x_output, y);

  // Expected values compared against those in "Numerical Example" in
  // https://optimization.cbe.cornell.edu/index.php?title=Conjugate_gradient_methods

  EXPECT_DOUBLE_EQ(h_x_output.at(0), 0.5641025641025641);
  EXPECT_DOUBLE_EQ(h_x_output.at(1), 0.17948717948717949);

  EXPECT_DOUBLE_EQ(y.at(0), 3.0);
  EXPECT_DOUBLE_EQ(y.at(1), 2.0); 
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BiconjugateGradientStabilizedTests, SolveWorksOnExample1)
{
  SetupBiconjugateGradientStabilizedExample1 setup {};

  BiconjugateGradientStabilized cg {
    setup.A_,
    setup.b_,
    setup.morphism_,
    setup.operations_};

  cg.create_default_initial_guess(setup.x_);

  const auto result = cg.solve(
    setup.x_,
    setup.Ax_,
    setup.r_,
    setup.p_,
    setup.s_);

  EXPECT_TRUE(get<0>(result));
  EXPECT_EQ(get<1>(result), 1);

  vector<double> h_x_output (setup.M_, 0.0);
  vector<double> y (setup.M_, 0.0);

  setup.x_.copy_device_output_to_host(h_x_output);

  setup.h_csr_.multiply(h_x_output, y);

  // Expected values compared against those in "Numerical Example" in
  // https://optimization.cbe.cornell.edu/index.php?title=Conjugate_gradient_methods

  EXPECT_DOUBLE_EQ(h_x_output.at(0), 1.0);
  EXPECT_DOUBLE_EQ(h_x_output.at(1), 2.0);
  EXPECT_DOUBLE_EQ(h_x_output.at(2), 1.0);
  EXPECT_DOUBLE_EQ(h_x_output.at(3), 2.0);

  EXPECT_DOUBLE_EQ(y.at(0), 0.0);
  EXPECT_DOUBLE_EQ(y.at(1), 6.0); 
  EXPECT_DOUBLE_EQ(y.at(2), 0.0);
  EXPECT_DOUBLE_EQ(y.at(3), 6.0); 
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(BiconjugateGradientStabilizedTests, SolveWorksOnExample2)
{
  SetupBiconjugateGradientStabilizedExample2 setup {};

  BiconjugateGradientStabilized cg {
    setup.A_,
    setup.b_,
    setup.morphism_,
    setup.operations_};

  cg.create_default_initial_guess(setup.x_);

  const auto result = cg.solve(
    setup.x_,
    setup.Ax_,
    setup.r_,
    setup.p_,
    setup.s_);

  EXPECT_TRUE(get<0>(result));
  EXPECT_EQ(get<1>(result), 0);

  vector<double> h_x_output (setup.M_, 0.0);
  vector<double> y (setup.M_, 0.0);

  setup.x_.copy_device_output_to_host(h_x_output);

  setup.h_csr_.multiply(h_x_output, y);

  // Expected values compared against those in "Numerical Example" in
  // https://optimization.cbe.cornell.edu/index.php?title=Conjugate_gradient_methods

  EXPECT_DOUBLE_EQ(h_x_output.at(0), 0.5);
  EXPECT_DOUBLE_EQ(h_x_output.at(1), 0.0);
  EXPECT_DOUBLE_EQ(h_x_output.at(2), -0.5);

  EXPECT_DOUBLE_EQ(y.at(0), 1.0);
  EXPECT_DOUBLE_EQ(y.at(1), 0.0); 
  EXPECT_DOUBLE_EQ(y.at(2), -1.0);
}

} // namespace Solvers
} // namespace Algebra
} // namespace GoogleUnitTests  
