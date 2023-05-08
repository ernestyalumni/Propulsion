#include "SparseMatrixMorphism.h"

#include "Algebra/Modules/Matrices/CompressedSparseRow.h"
#include "Utilities/HandleUnsuccessfulCuSparseCall.h"
#include "Utilities/HandleUnsuccessfulCudaCall.h"

#include <cusparse.h> // cusparseSpMatDescr_t

using Algebra::Modules::Matrices::SparseMatrices::CompressedSparseRowMatrix;
using Utilities::HandleUnsuccessfulCUDACall;
using Utilities::HandleUnsuccessfulCuSparseCall;

namespace Algebra
{
namespace Modules
{
namespace Morphisms
{

SparseMatrixMorphismOnDenseVectors::SparseMatrixMorphismOnDenseVectors(
  const float alpha,
  const float beta
  ):
  buffer_{nullptr},
  cusparse_handle_{0},
  buffer_size_{0},
  alpha_{alpha},
  beta_{beta}
{
  HandleUnsuccessfulCuSparseCall handle_create_handle {
    "Failed to create cuSparse handle"};

  // https://docs.nvidia.com/cuda/cusparse/index.html#cusparsecreate
  // cusparseStatus_t cusparseCreate(cusparseHandle_t* handle) initializes
  // cuSparse library and creates handle on cuSparse context. It allocates
  // hardware resources necessary for accessing GPU.
  handle_create_handle(cusparseCreate(&cusparse_handle_));
}

SparseMatrixMorphismOnDenseVectors::~SparseMatrixMorphismOnDenseVectors()
{
  HandleUnsuccessfulCuSparseCall handle_destroy_handle {
    "Failed to destroy cuSparse handle"};

  // ref: https://docs.nvidia.com/cuda/cusparse/index.html#cusparsedestroy
  // Releases CPU-side resources used by cuSparse library. Release of GPU-side
  // resources maybe deferred until application shuts down.
  handle_destroy_handle(cusparseDestroy(cusparse_handle_));    

  HandleUnsuccessfulCUDACall handle_free_buffer {"Failed to free buffer"};

  handle_free_buffer(cudaFree(buffer_));
}

bool SparseMatrixMorphismOnDenseVectors::linear_transform(
  CompressedSparseRowMatrix& A,
  DenseVector& x,
  DenseVector& y)
{
  HandleUnsuccessfulCuSparseCall handle_multiplication {
    "Failed to multiply with sparse matrix and dense vector"};

  // Performs multiplication of a sparse matrix and dense vector.

  handle_multiplication(cusparseSpMV(
    cusparse_handle_,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha_,
    A.matrix_descriptor_,
    x.vector_descriptor_,
    &beta_,
    y.vector_descriptor_,
    CUDA_R_32F,
    CUSPARSE_SPMV_ALG_DEFAULT,
    buffer_));

  return handle_multiplication.is_cusparse_success();
}

bool SparseMatrixMorphismOnDenseVectors::buffer_size(
  CompressedSparseRowMatrix& A,
  DenseVector& x,
  DenseVector& y)
{
  HandleUnsuccessfulCuSparseCall handle_buffer_size {"Failed to buffer size"};

  // See https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespmv
  handle_buffer_size(cusparseSpMV_bufferSize(
    cusparse_handle_,
    // op(A) = A for CUSPARSE_OPERATION_NON_TRANSPOSE.
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha_,
    A.matrix_descriptor_,
    x.vector_descriptor_,
    &beta_,
    y.vector_descriptor_,
    // cudaDataType computeType; mixed regular, complex computation.
    CUDA_R_32F,
    // cusparseSpMVAlg_t alg - algorithm for computation.
    CUSPARSE_SPMV_ALG_DEFAULT,
    // void* externalBuffer - pointer to a workspace buffer of at least
    // bufferSize bytes.
    &buffer_size_));

  HandleUnsuccessfulCUDACall handle_allocate_buffer {
    "Failed to allocate for buffer"};

  handle_allocate_buffer(cudaMalloc(&buffer_, buffer_size_));

  return handle_buffer_size.is_cusparse_success() &&
    handle_allocate_buffer.is_cuda_success();
}

} // namespace Morphisms
} // namespace Modules
} // namespace Algebra
