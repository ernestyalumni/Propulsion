#include "Algebra/Modules/Matrices/CompressedSparseRow.h"
#include "Algebra/Modules/Matrices/GenerateCompressedSparseRowMatrix.h"
#include "Algebra/Modules/Matrices/HostCompressedSparseRow.h"
#include "Algebra/Modules/Morphisms/SparseMatrixMorphism.h"
#include "Algebra/Modules/Vectors/CuBLASVectorOperations.h"
#include "Algebra/Solvers/ConjugateGradient.h"
#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <tuple> // std::get
#include <vector>

using Algebra::Modules::Matrices::SparseMatrices::CompressedSparseRowMatrix;
using Algebra::Modules::Matrices::SparseMatrices::DenseVector;
using Algebra::Modules::Matrices::SparseMatrices::HostCompressedSparseRowMatrix;
using Algebra::Modules::Matrices::SparseMatrices::generate_tridiagonal_matrix;
using Algebra::Modules::Morphisms::SparseMatrixMorphismOnDenseVector;
using Algebra::Modules::Vectors::Array;
using Algebra::Modules::Vectors::CuBLASVectorOperations;
using Algebra::Solvers::ConjugateGradient;
using Algebra::Solvers::SampleConjugateGradient;
using std::size_t;
using std::vector;

namespace GoogleUnitTests
{
namespace Algebra
{
namespace Solvers
{

struct SetupSampleConjugateGradientTests
{
  static constexpr size_t M_ {1048576};
  static constexpr size_t number_of_nonzero_elements {
    (M_ - 2) * 3 + 4};

  HostCompressedSparseRowMatrix h_csr_;
  CompressedSparseRowMatrix A_;
  DenseVector x_;
  DenseVector Ax_;
  SparseMatrixMorphismOnDenseVector morphism_;
  SampleConjugateGradient cg_;
  Array r_;
  CuBLASVectorOperations operations_;

  SetupSampleConjugateGradientTests(
    const float alpha=1.0f,
    const float beta=0.0f
    ):
    h_csr_{M_, M_, number_of_nonzero_elements},
    A_{M_, M_, number_of_nonzero_elements},
    x_{M_},
    Ax_{M_},
    morphism_{alpha, beta},
    cg_{},
    r_{M_},
    operations_{}
  {
    generate_tridiagonal_matrix(h_csr_);
    A_.copy_host_input_to_device(h_csr_);
    cg_.create_default_initial_guess(x_);

    morphism_.buffer_size(A_, x_, Ax_);

    vector<float> h_r (M_, 1.0f);
    r_.copy_host_input_to_device(h_r);
  }
};

struct SetupConjugateGradientTests
{
  static constexpr size_t M_ {1048576};
  static constexpr size_t number_of_nonzero_elements {
    (M_ - 2) * 3 + 4};

  HostCompressedSparseRowMatrix h_csr_;
  CompressedSparseRowMatrix A_;
  DenseVector b_;
  DenseVector x_;
  DenseVector Ax_;
  SparseMatrixMorphismOnDenseVector morphism_;
  Array r_;
  CuBLASVectorOperations operations_;

  SetupConjugateGradientTests(
    const float alpha=1.0f,
    const float beta=0.0f
    ):
    h_csr_{M_, M_, number_of_nonzero_elements},
    A_{M_, M_, number_of_nonzero_elements},
    b_{M_},
    x_{M_},
    Ax_{M_},
    morphism_{alpha, beta},
    r_{M_},
    operations_{}
  {
    generate_tridiagonal_matrix(h_csr_);
    A_.copy_host_input_to_device(h_csr_);

    morphism_.buffer_size(A_, x_, Ax_);

    vector<float> h_r (M_, 1.0f);
    r_.copy_host_input_to_device(h_r);
    b_.copy_host_input_to_device(h_r);
  }
};

// See "Numerical Example" of
// https://optimization.cbe.cornell.edu/index.php?title=Conjugate_gradient_methods
struct SetupConjugateGradientExample
{
  static constexpr size_t M_ {2};
  static constexpr size_t number_of_nonzero_elements {4};

  HostCompressedSparseRowMatrix h_csr_;
  CompressedSparseRowMatrix A_;
  DenseVector b_;
  DenseVector x_;
  DenseVector Ax_;
  SparseMatrixMorphismOnDenseVector morphism_;
  Array r_;
  CuBLASVectorOperations operations_;

  SetupConjugateGradientExample(
    const float alpha=1.0f,
    const float beta=0.0f
    ):
    h_csr_{M_, M_, number_of_nonzero_elements},
    A_{M_, M_, number_of_nonzero_elements},
    b_{M_},
    x_{M_},
    Ax_{M_},
    morphism_{alpha, beta},
    r_{M_},
    operations_{}
  {
    h_csr_.copy_values(vector<float>{5, 1, 1, 8});
    h_csr_.copy_row_offsets(vector<int>{0, 2, 4});
    h_csr_.copy_column_indices(vector<int>{0, 1, 0, 1});

    A_.copy_host_input_to_device(h_csr_);

    morphism_.buffer_size(A_, x_, Ax_);

    vector<float> h_b {3.0, 2.0};

    b_.copy_host_input_to_device(h_b);
  }
};

struct SetupConjugateGradientExample1
{
  static constexpr size_t M_ {4};
  static constexpr size_t number_of_nonzero_elements {12};

  HostCompressedSparseRowMatrix h_csr_;
  CompressedSparseRowMatrix A_;
  DenseVector b_;
  DenseVector x_;
  DenseVector Ax_;
  SparseMatrixMorphismOnDenseVector morphism_;
  Array r_;
  CuBLASVectorOperations operations_;

  SetupConjugateGradientExample1(
    const float alpha=1.0f,
    const float beta=0.0f
    ):
    h_csr_{M_, M_, number_of_nonzero_elements},
    A_{M_, M_, number_of_nonzero_elements},
    b_{M_},
    x_{M_},
    Ax_{M_},
    morphism_{alpha, beta},
    r_{M_},
    operations_{}
  {
    h_csr_.copy_values(
      vector<float>{4, -1, -1, -1, 4, -1, -1, 4, -1, -1, -1, 4});
    h_csr_.copy_row_offsets(vector<int>{0, 3, 6, 9, 12});
    h_csr_.copy_column_indices(vector<int>{0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3});

    A_.copy_host_input_to_device(h_csr_);

    morphism_.buffer_size(A_, x_, Ax_);

    vector<float> h_b {0, 6, 0, 6};

    b_.copy_host_input_to_device(h_b);
  }
};

struct SetupConjugateGradientExample2
{
  static constexpr size_t M_ {3};
  static constexpr size_t number_of_nonzero_elements {7};

  HostCompressedSparseRowMatrix h_csr_;
  CompressedSparseRowMatrix A_;
  DenseVector b_;
  DenseVector x_;
  DenseVector Ax_;
  SparseMatrixMorphismOnDenseVector morphism_;
  Array r_;
  CuBLASVectorOperations operations_;

  SetupConjugateGradientExample2(
    const float alpha=1.0f,
    const float beta=0.0f
    ):
    h_csr_{M_, M_, number_of_nonzero_elements},
    A_{M_, M_, number_of_nonzero_elements},
    b_{M_},
    x_{M_},
    Ax_{M_},
    morphism_{alpha, beta},
    r_{M_},
    operations_{}
  {
    h_csr_.copy_values(
      vector<float>{2, -1, -1, 2, -1, -1, 2});
    h_csr_.copy_row_offsets(vector<int>{0, 2, 5, 7});
    h_csr_.copy_column_indices(vector<int>{0, 1, 0, 1, 2, 1, 2});

    A_.copy_host_input_to_device(h_csr_);

    morphism_.buffer_size(A_, x_, Ax_);

    vector<float> h_b {1, 0, -1};

    b_.copy_host_input_to_device(h_b);
  }
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SampleConjugateGradientTests, ConstructibleWithDefaultValues)
{
  SampleConjugateGradient cg {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SampleConjugateGradientTests, CreateDefaultInitialGuessCreates0)
{
  SampleConjugateGradient cg {};

  DenseVector x {69};

  cg.create_default_initial_guess(x);

  std::array<float, 69> h_output {};

  x.copy_device_output_to_host(h_output);

  for (size_t i {0}; i < 69; ++i)
  {
    EXPECT_FLOAT_EQ(h_output.at(i), 0.0f);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SampleConjugateGradientTests, InitialStepWorks)
{
  SetupSampleConjugateGradientTests setup {};
  const auto result = setup.cg_.initial_step(
    setup.morphism_,
    setup.operations_,
    setup.A_,
    setup.x_,
    setup.Ax_,
    setup.r_);

  EXPECT_TRUE(result.has_value());
  EXPECT_FLOAT_EQ(*result, 1048576.0f);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SampleConjugateGradientTests, StepWorksAfterInitialStep)
{
  SetupSampleConjugateGradientTests setup {};
  const auto r_0_sqrt = setup.cg_.initial_step(
    setup.morphism_,
    setup.operations_,
    setup.A_,
    setup.x_,
    setup.Ax_,
    setup.r_);

  ASSERT_TRUE(r_0_sqrt.has_value());

  float r_1 {*r_0_sqrt};
  float r_0 {0.0f};

  DenseVector p {setup.M_};

  const auto result = setup.cg_.step(
    1,
    r_0,
    r_1,
    p,
    setup.r_,
    setup.morphism_,
    setup.operations_,
    setup.A_,
    setup.Ax_,
    setup.x_);

  EXPECT_TRUE(result.has_value());
  EXPECT_FLOAT_EQ(std::get<0>(*result), 1048576);
  EXPECT_FLOAT_EQ(std::get<1>(*result), 1985.8337);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SampleConjugateGradientTests, StepWorksAfterFirstStep)
{
  SetupSampleConjugateGradientTests setup {};
  const auto r_0_sqrt = setup.cg_.initial_step(
    setup.morphism_,
    setup.operations_,
    setup.A_,
    setup.x_,
    setup.Ax_,
    setup.r_);

  float r_1 {*r_0_sqrt};
  float r_0 {0.0f};

  DenseVector p {setup.M_};

  const auto r_0_r_1_0 = setup.cg_.step(
    1,
    r_0,
    r_1,
    p,
    setup.r_,
    setup.morphism_,
    setup.operations_,
    setup.A_,
    setup.Ax_,
    setup.x_);

  EXPECT_TRUE(r_0_r_1_0.has_value());
  r_0 = std::get<0>(*r_0_r_1_0);
  r_1 = std::get<1>(*r_0_r_1_0);

  const auto r_0_r_1_1 = setup.cg_.step(
    2,
    r_0,
    r_1,
    p,
    setup.r_,
    setup.morphism_,
    setup.operations_,
    setup.A_,
    setup.Ax_,
    setup.x_);

  EXPECT_TRUE(r_0_r_1_1.has_value());
  EXPECT_FLOAT_EQ(std::get<0>(*r_0_r_1_1), 1981.6189);
  EXPECT_FLOAT_EQ(std::get<1>(*r_0_r_1_1), 10.547638);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SampleConjugateGradientTests, SolveWorks)
{
  SetupSampleConjugateGradientTests setup {};
  DenseVector p {setup.M_};

  EXPECT_TRUE(setup.cg_.solve(
    p,
    setup.r_,
    setup.morphism_,
    setup.operations_,
    setup.A_,
    setup.Ax_,
    setup.x_));  

  vector<float> h_x_output (setup.M_, 0.0f);
  vector<float> y (setup.M_, 0.0f);

  setup.x_.copy_device_output_to_host(h_x_output);

  setup.h_csr_.multiply(h_x_output, y);

  for (std::size_t i {0}; i < setup.M_; ++i)
  {
    EXPECT_TRUE(std::abs(y.at(i) - 1.0f) < 1e-6f);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ConjugateGradientTests, Constructible)
{
  SetupConjugateGradientTests setup {};

  ConjugateGradient cg {setup.A_, setup.b_, setup.morphism_, setup.operations_};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ConjugateGradientTests, CreateDefaultInitialGuessCreates0)
{
  SetupConjugateGradientTests setup {};

  ConjugateGradient cg {setup.A_, setup.b_, setup.morphism_, setup.operations_};

  DenseVector x {69};

  cg.create_default_initial_guess(x);

  std::array<float, 69> h_output {};

  x.copy_device_output_to_host(h_output);

  for (size_t i {0}; i < 69; ++i)
  {
    EXPECT_FLOAT_EQ(h_output.at(i), 0.0f);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ConjugateGradientTests, InitialStepWorks)
{
  SetupConjugateGradientTests setup {};

  ConjugateGradient cg {setup.A_, setup.b_, setup.morphism_, setup.operations_};

  cg.create_default_initial_guess(setup.x_);

  const auto result = cg.initial_step(setup.x_, setup.Ax_, setup.r_);

  EXPECT_TRUE(result.has_value());
  EXPECT_FLOAT_EQ(*result, 1048576.0f);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ConjugateGradientTests, StepWorksAfterInitialStep)
{
  SetupConjugateGradientTests setup {};

  ConjugateGradient cg {setup.A_, setup.b_, setup.morphism_, setup.operations_};

  cg.create_default_initial_guess(setup.x_);

  const auto r_0_sqrt = cg.initial_step(setup.x_, setup.Ax_, setup.r_);

  ASSERT_TRUE(r_0_sqrt.has_value());

  float r_1 {*r_0_sqrt};
  float r_0 {0.0f};

  DenseVector p {setup.M_};

  const auto result = cg.step(
    0,
    r_0,
    r_1,
    setup.x_,
    setup.r_,
    p,
    setup.Ax_);

  EXPECT_TRUE(result.has_value());
  EXPECT_FLOAT_EQ(std::get<0>(*result), 1048576);
  EXPECT_FLOAT_EQ(std::get<1>(*result), 1979.2708);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ConjugateGradientTests, StepWorksAfterFirstStep)
{
  SetupConjugateGradientTests setup {};
  ConjugateGradient cg {setup.A_, setup.b_, setup.morphism_, setup.operations_};
  cg.create_default_initial_guess(setup.x_);

  const auto r_0_sqrt = cg.initial_step(setup.x_, setup.Ax_, setup.r_);

  float r_1 {*r_0_sqrt};
  float r_0 {0.0f};

  DenseVector p {setup.M_};

  const auto r_0_r_1_0 = cg.step(
    0,
    r_0,
    r_1,
    setup.x_,
    setup.r_,
    p,
    setup.Ax_);

  EXPECT_TRUE(r_0_r_1_0.has_value());
  r_0 = std::get<0>(*r_0_r_1_0);
  r_1 = std::get<1>(*r_0_r_1_0);

  const auto r_0_r_1_1 = cg.step(
    1,
    r_0,
    r_1,
    setup.x_,
    setup.r_,
    p,
    setup.Ax_);

  EXPECT_TRUE(r_0_r_1_1.has_value());
  EXPECT_FLOAT_EQ(std::get<0>(*r_0_r_1_1), 1982.2006);
  EXPECT_FLOAT_EQ(std::get<1>(*r_0_r_1_1), 10.574238);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ConjugateGradientTests, SolveWorks)
{
  SetupConjugateGradientTests setup {};
  ConjugateGradient cg {setup.A_, setup.b_, setup.morphism_, setup.operations_};
  cg.create_default_initial_guess(setup.x_);

  DenseVector p {setup.M_};

  const auto result = cg.solve(setup.x_, setup.Ax_, setup.r_, p);

  EXPECT_TRUE(std::get<0>(result));
  EXPECT_EQ(std::get<1>(result), 8);

  vector<float> h_x_output (setup.M_, 0.0f);
  vector<float> y (setup.M_, 0.0f);

  setup.x_.copy_device_output_to_host(h_x_output);

  setup.h_csr_.multiply(h_x_output, y);

  for (std::size_t i {0}; i < setup.M_; ++i)
  {
    EXPECT_TRUE(std::abs(y.at(i) - 1.0f) < 1e-6f);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ConjugateGradientTests, SolveWorksOnExample)
{
  SetupConjugateGradientExample setup {};
  ConjugateGradient cg {setup.A_, setup.b_, setup.morphism_, setup.operations_};
  cg.create_default_initial_guess(setup.x_);

  DenseVector p {setup.M_};

  const auto result = cg.solve(setup.x_, setup.Ax_, setup.r_, p);

  EXPECT_TRUE(std::get<0>(result));
  EXPECT_EQ(std::get<1>(result), 2);

  vector<float> h_x_output (setup.M_, 0.0f);
  vector<float> y (setup.M_, 0.0f);

  setup.x_.copy_device_output_to_host(h_x_output);

  setup.h_csr_.multiply(h_x_output, y);

  // Expected values compared against those in "Numerical Example" in
  // https://optimization.cbe.cornell.edu/index.php?title=Conjugate_gradient_methods

  EXPECT_FLOAT_EQ(h_x_output.at(0), 0.56410259);
  EXPECT_FLOAT_EQ(h_x_output.at(1), 0.17948718);

  EXPECT_FLOAT_EQ(y.at(0), 3.0);
  EXPECT_FLOAT_EQ(y.at(1), 2.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ConjugateGradientTests, SolveWorksOnExample1)
{
  SetupConjugateGradientExample1 setup {};
  ConjugateGradient cg {setup.A_, setup.b_, setup.morphism_, setup.operations_};
  cg.create_default_initial_guess(setup.x_);

  DenseVector p {setup.M_};

  const auto result = cg.solve(setup.x_, setup.Ax_, setup.r_, p);

  EXPECT_TRUE(std::get<0>(result));
  EXPECT_EQ(std::get<1>(result), 2);

  vector<float> h_x_output (setup.M_, 0.0f);
  vector<float> y (setup.M_, 0.0f);

  setup.x_.copy_device_output_to_host(h_x_output);

  setup.h_csr_.multiply(h_x_output, y);

  EXPECT_FLOAT_EQ(h_x_output.at(0), 1);
  EXPECT_FLOAT_EQ(h_x_output.at(1), 2);
  EXPECT_FLOAT_EQ(h_x_output.at(2), 1);
  EXPECT_FLOAT_EQ(h_x_output.at(3), 2);

  EXPECT_FLOAT_EQ(y.at(0), 0.0);
  EXPECT_FLOAT_EQ(y.at(1), 6.0);
  EXPECT_FLOAT_EQ(y.at(2), 0.0);
  EXPECT_FLOAT_EQ(y.at(3), 6.0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ConjugateGradientTests, SolveWorksOnExample2)
{
  SetupConjugateGradientExample2 setup {};
  ConjugateGradient cg {setup.A_, setup.b_, setup.morphism_, setup.operations_};
  cg.create_default_initial_guess(setup.x_);

  DenseVector p {setup.M_};

  const auto result = cg.solve(setup.x_, setup.Ax_, setup.r_, p);

  EXPECT_TRUE(std::get<0>(result));
  EXPECT_EQ(std::get<1>(result), 1);

  vector<float> h_x_output (setup.M_, 0.0f);
  vector<float> y (setup.M_, 0.0f);

  setup.x_.copy_device_output_to_host(h_x_output);

  setup.h_csr_.multiply(h_x_output, y);

  EXPECT_FLOAT_EQ(h_x_output.at(0), 0.5);
  EXPECT_FLOAT_EQ(h_x_output.at(1), 0);
  EXPECT_FLOAT_EQ(h_x_output.at(2), -0.5);

  EXPECT_FLOAT_EQ(y.at(0), 1.0);
  EXPECT_FLOAT_EQ(y.at(1), 0.0);
  EXPECT_FLOAT_EQ(y.at(2), -1.0);
}

} // namespace Solvers
} // namespace Algebra
} // namespace GoogleUnitTests  
