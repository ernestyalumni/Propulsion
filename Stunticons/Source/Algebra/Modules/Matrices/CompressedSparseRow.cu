#include "CompressedSparseRow.h"

#include "Algebra/Modules/Vectors/HostArrays.h"
#include "HostCompressedSparseRow.h"
#include "Utilities/HandleUnsuccessfulCuSparseCall.h"
#include "Utilities/HandleUnsuccessfulCudaCall.h"

#include <cstddef> // std::size_t
#include <cuda_runtime.h> // cudaFree, cudaMalloc
#include <cusparse.h> // cuSparseCreateCsr
#include <iostream> // std::cerr

using Algebra::Modules::Vectors::HostArray;
using Utilities::HandleUnsuccessfulCUDACall;
using Utilities::HandleUnsuccessfulCuSparseCall;
using std::cerr;
using std::size_t;

namespace Algebra
{
namespace Modules
{
namespace Matrices
{
namespace SparseMatrices
{

CompressedSparseRowMatrix::CompressedSparseRowMatrix(
  const size_t M,
  const size_t N,
  const size_t number_of_elements
  ):
  d_values_{nullptr},
  d_columns_{nullptr},
  d_rows_{nullptr},
  M_{M},
  N_{N},
  number_of_elements_{number_of_elements},
  matrix_handler_{nullptr}  
{
  const cudaError_t err_values {
    cudaMalloc(
      reinterpret_cast<void**>(&d_values_),
      number_of_elements_ * sizeof(float))};
  const cudaError_t err_columns {
    cudaMalloc(
      reinterpret_cast<void**>(&d_columns_),
      number_of_elements_ * sizeof(int))};
  const cudaError_t err_rows {
    cudaMalloc(
      reinterpret_cast<void**>(&d_rows_),
      (M + 1) * sizeof(int))};

  if (err_values != cudaSuccess)
  {
    cerr << "Failed to allocate device array for values (error code " <<
      cudaGetErrorString(err_values) << ")!\n";
  }  

  if (err_columns != cudaSuccess)
  {
    cerr << "Failed to allocate device array for column indices (error code " <<
      cudaGetErrorString(err_columns) << ")!\n";
  }  

  if (err_rows != cudaSuccess)
  {
    cerr << "Failed to allocate device array for row indices (error code " <<
      cudaGetErrorString(err_rows) << ")!\n";
  }  

  // ref. https://docs.nvidia.com/cuda/cusparse/index.html#cusparsecreatecsr
  // cusparseCreateCsr initializes sparse matrix descriptor cusparseSpMatDescr_t
  // spMatDescr in CSR format.
  const cusparseStatus_t create_sparse_status {cusparseCreateCsr(
    &matrix_handler_,
    // number of rows
    M,
    // number of columns
    N,
    // Number of non-zero entries of sparse matrix.
    number_of_elements_,
    // Row offsets of sparse matrix.
    d_rows_,
    d_columns_,
    d_values_,
    // Data type of csrRowOffsets.
    CUSPARSE_INDEX_32I,
    // Data type of csrColInd.
    CUSPARSE_INDEX_32I,
    // Index base of csrRowOffsets and csrColInd.
    CUSPARSE_INDEX_BASE_ZERO,
    // Datatype of csrValues.
    CUDA_R_32F)};

  if (create_sparse_status != CUSPARSE_STATUS_SUCCESS)
  {
    cerr << "Failed to create Sparse CSR (error code " <<
      // ref. https://docs.nvidia.com/cuda/cusparse/index.html#cusparsegeterrorstring
      // const char* cusparseGetErrorSTring(cusparseStatus_t status).
      cusparseGetErrorString(create_sparse_status) << ")!\n";    
  }
}

CompressedSparseRowMatrix::~CompressedSparseRowMatrix()
{
  const cudaError_t err_values {cudaFree(d_values_)};
  const cudaError_t err_columns {cudaFree(d_columns_)};
  const cudaError_t err_rows {cudaFree(d_rows_)};

  if (err_values != cudaSuccess)
  {
    cerr << "Failed to free device array for values (error code " <<
      cudaGetErrorString(err_values) << ")!\n";
  }
  if (err_columns != cudaSuccess)
  {
    cerr << "Failed to free device array for column indicies (error code " <<
      cudaGetErrorString(err_columns) << ")!\n";
  }
  if (err_rows != cudaSuccess)
  {
    cerr << "Failed to free device array for row indices (error code " <<
      cudaGetErrorString(err_rows) << ")!\n";
  }

  if (matrix_handler_)
  {
    // ref. https://docs.nvidia.com/cuda/cusparse/index.html#cusparsedestroyspmat
    // Releases host memory allocated for sparse matrix descriptor spMatDescr.
    const cusparseStatus_t destroy_matrix_status {cusparseDestroySpMat(
      matrix_handler_)};

    if (destroy_matrix_status != CUSPARSE_STATUS_SUCCESS)
    {
      cerr << "Failed to destroy Sparse CSR descriptor (error code " <<
        cusparseGetErrorString(destroy_matrix_status) << ")!\n";    
    }
  }

  // We choose not to throw upon a failed garbage clean up.
}

void CompressedSparseRowMatrix::copy_host_input_to_device(
  const HostCompressedSparseRowMatrix& h_a)
{
  HandleUnsuccessfulCUDACall handle_columns {
    "Failed to copy column indices from host to device"};

  handle_columns(cudaMemcpy(
    d_columns_,
    h_a.J_,
    h_a.number_of_elements_ * sizeof(int),
    cudaMemcpyHostToDevice));

  HandleUnsuccessfulCUDACall handle_rows {
    "Failed to copy row from host to device"};

  handle_rows(cudaMemcpy(
    d_rows_,
    h_a.I_,
    (h_a.M_ + 1) * sizeof(int),
    cudaMemcpyHostToDevice));

  HandleUnsuccessfulCUDACall handle_values {
    "Failed to copy values from host to device"};

  handle_values(cudaMemcpy(
    d_values_,
    h_a.values_,
    h_a.number_of_elements_ * sizeof(float),
    cudaMemcpyHostToDevice));
}

void CompressedSparseRowMatrix::copy_device_output_to_host(
  HostCompressedSparseRowMatrix& h_a)
{
  HandleUnsuccessfulCUDACall handle_columns {
    "Failed to copy column indices from device to host"};

  handle_columns(cudaMemcpy(
    h_a.J_,
    d_columns_,
    number_of_elements_ * sizeof(int),
    cudaMemcpyDeviceToHost));

  HandleUnsuccessfulCUDACall handle_rows {
    "Failed to copy row from device to host"};

  handle_rows(cudaMemcpy(
    h_a.I_,
    d_rows_,
    (M_ + 1) * sizeof(int),
    cudaMemcpyDeviceToHost));

  HandleUnsuccessfulCUDACall handle_values {
    "Failed to copy values from device to host"};

  handle_values(cudaMemcpy(
    h_a.values_,
    d_values_,
    number_of_elements_ * sizeof(float),
    cudaMemcpyDeviceToHost));  
}

DenseVector::DenseVector(
  const size_t N
  ):
  d_values_{nullptr},
  number_of_elements_{N},
  vector_descriptor_{nullptr}  
{
  HandleUnsuccessfulCUDACall handle_cuda_malloc {
    "Failed to allocate device array for values"};

  handle_cuda_malloc(
    cudaMalloc(
      reinterpret_cast<void**>(&d_values_),
      number_of_elements_ * sizeof(float)));

  // ref. https://docs.nvidia.com/cuda/cusparse/index.html#cusparsecreatednvec
  // Initializes dense vector descriptor.
  const cusparseStatus_t create_dense_vector_status {
    cusparseCreateDnVec(
      &vector_descriptor_,
      number_of_elements_,
      d_values_,
      CUDA_R_32F)};

  if (create_dense_vector_status != CUSPARSE_STATUS_SUCCESS)
  {
    cerr << "Failed to create dense vector (error code " <<
      cusparseGetErrorString(create_dense_vector_status) << ")!\n";
  }
}

DenseVector::~DenseVector()
{
  const cudaError_t err_values {cudaFree(d_values_)};

  HandleUnsuccessfulCUDACall handle_cuda_free {
    "Failed to free device array for values"};

  handle_cuda_free(err_values);

  if (vector_descriptor_)
  {
    HandleUnsuccessfulCuSparseCall handle_destroy_dense_vector {
      "Failed to destroy dense vector"};
    
    handle_destroy_dense_vector(cusparseDestroyDnVec(vector_descriptor_));
  }
}

void DenseVector::copy_host_input_to_device(const HostArray& h_a)
{
  HandleUnsuccessfulCUDACall handle_values {
    "Failed to copy values from host to device"};

  handle_values(cudaMemcpy(
    d_values_,
    h_a.values_,
    h_a.number_of_elements_ * sizeof(float),
    cudaMemcpyHostToDevice));
}

void DenseVector::copy_device_output_to_host(HostArray& h_a)
{
  HandleUnsuccessfulCUDACall handle_values {
    "Failed to copy values from device to host"};

  handle_values(cudaMemcpy(
    h_a.values_,
    d_values_,
    number_of_elements_ * sizeof(float),
    cudaMemcpyDeviceToHost));  
}

} // namespace SparseMatrices
} // namespace Matrices
} // namespace Modules
} // namespace Algebra
