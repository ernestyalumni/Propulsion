#include "Algebra/Modules/Matrices/CompressedSparseRow.h"
#include "Algebra/Modules/Matrices/GenerateCompressedSparseRowMatrix.h"
#include "Algebra/Modules/Matrices/HostCompressedSparseRow.h"
#include "Algebra/Modules/Morphisms/SparseMatrixMorphism.h"
#include "Algebra/Modules/Vectors/CuBLASVectorOperations.h"
#include "Optimization/ConjugateGradient.h"
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
using Optimization::ConjugateGradient;
using std::size_t;
using std::vector;

namespace GoogleUnitTests
{
namespace Optimization
{

struct SetupConjugateGradientTests
{
  static constexpr size_t M_ {1048576};
  static constexpr size_t number_of_nonzero_elements {
    (M_ - 2) * 3 + 4};

  HostCompressedSparseRowMatrix h_csr_;
  CompressedSparseRowMatrix A_;
  DenseVector x_;
  DenseVector Ax_;
  SparseMatrixMorphismOnDenseVector morphism_;
  ConjugateGradient cg_;
  Array r_;
  CuBLASVectorOperations operations_;

  SetupConjugateGradientTests(const float alpha=1.0f, const float beta=0.0f):
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

    std::vector<float> h_r (M_, 1.0f);
    r_.copy_host_input_to_device(h_r);
  }
};

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ConjugateGradientTests, ConstructibleWithDefaultValues)
{
  ConjugateGradient cg {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ConjugateGradientTests, CreateDefaultInitialGuessCreates0)
{
  ConjugateGradient cg {};

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
TEST(ConjugateGradientTests, StepWorksAfterInitialStep)
{
  SetupConjugateGradientTests setup {};
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
TEST(ConjugateGradientTests, StepWorksAfterFirstStep)
{
  SetupConjugateGradientTests setup {};
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

  EXPECT_TRUE(r_0_r_1_1.has_value());
  EXPECT_FLOAT_EQ(std::get<0>(*r_0_r_1_1), 1981.6189);
  EXPECT_FLOAT_EQ(std::get<1>(*r_0_r_1_1), 14.725352);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(ConjugateGradientTests, SolveWorks)
{
  SetupConjugateGradientTests setup {};
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

} // namespace Optimization
} // namespace GoogleUnitTests  
