#include "Array.h"
#include "CuBLASVectorOperations.h"
#include "Utilities/HandleUnsuccessfulCuBLASCall.h"
#include "cublas_v2.h"

#include <optional>

using Utilities::HandleUnsuccessfulCuBLASCall;
using std::make_optional;
using std::nullopt;
using std::optional;

namespace Algebra
{
namespace Modules
{
namespace Vectors
{

CuBLASVectorOperations::CuBLASVectorOperations():
  cublas_handle_{0}
{
  HandleUnsuccessfulCuBLASCall handle_create_handle {
    "Failed to create cuBLAS handle"};

  // ref. https://docs.nvidia.com/cuda/cublas/index.html#cublascreate
  // Initializes cuBLAS library and creates handle to an opaque structure
  // holding cuBLAS library context. It allocated hardware resources on the host
  // and device and must be called prior to making any other cuBLAS library
  // calls. cuBLAS library context is tied to current CUDA device.
  handle_create_handle(cublasCreate(&cublas_handle_));
}

CuBLASVectorOperations::~CuBLASVectorOperations()
{
  HandleUnsuccessfulCuBLASCall handle_destroy_handle {
    "Failed to destroy cuBLAS handle"};

  handle_destroy_handle(cublasDestroy(cublas_handle_));
}

bool CuBLASVectorOperations::scalar_multiply_and_add_vector(
  const float input_alpha,
  const DenseVector& x,
  Array& y)
{
  HandleUnsuccessfulCuBLASCall handle_vector_operations {
    "Failed to scalar multiple and add in cuBLAS"};

  const float alpha {input_alpha};

  // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-axpy  
  // Multiplies vector x by the scalar alpha and adds it to vector y overwriting
  // latest vector with result.
  // S in Saxpy stands for single-precision floating point.
  handle_vector_operations(cublasSaxpy(
    cublas_handle_,
    x.number_of_elements_,
    &alpha,
    // const float* x, vector with n elements.
    x.d_values_,
    // int incx - stride between consecutive elements of x.
    1,
    y.values_,
    // int incy - stride between consecutive elements of y.
    1));

  return handle_vector_operations.is_cuBLAS_success();
}

bool CuBLASVectorOperations::scalar_multiply_and_add_vector(
  const float input_alpha,
  const DenseVector& x,
  DenseVector& y)
{
  HandleUnsuccessfulCuBLASCall handle_vector_operations {
    "Failed to scalar multiple and add in cuBLAS"};

  const float alpha {input_alpha};

  handle_vector_operations(cublasSaxpy(
    cublas_handle_,
    x.number_of_elements_,
    &alpha,
    // const float* x, vector with n elements.
    x.d_values_,
    // int incx - stride between consecutive elements of x.
    1,
    y.d_values_,
    // int incy - stride between consecutive elements of y.
    1));

  return handle_vector_operations.is_cuBLAS_success();
}

bool CuBLASVectorOperations::scalar_multiply_and_add_vector(
  const float input_alpha,
  const Array& x,
  Array& y)
{
  HandleUnsuccessfulCuBLASCall handle_vector_operations {
    "Failed to scalar multiply and add in cuBLAS"};

  const float alpha {input_alpha};

  // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-axpy  
  // Multiplies vector x by the scalar alpha and adds it to vector y overwriting
  // latest vector with result.
  // S in Saxpy stands for single-precision floating point.
  handle_vector_operations(cublasSaxpy(
    cublas_handle_,
    x.number_of_elements_,
    &alpha,
    // const float* x, vector with n elements.
    x.values_,
    // int incx - stride between consecutive elements of x.
    1,
    y.values_,
    // int incy - stride between consecutive elements of y.
    1));

  return handle_vector_operations.is_cuBLAS_success();
}

bool CuBLASVectorOperations::scalar_multiply_and_add_vector(
  const float input_alpha,
  const Array& x,
  DenseVector& y)
{
  HandleUnsuccessfulCuBLASCall handle_vector_operations {
    "Failed to scalar multiple and add in cuBLAS"};

  const float alpha {input_alpha};

  handle_vector_operations(cublasSaxpy(
    cublas_handle_,
    x.number_of_elements_,
    &alpha,
    x.values_,
    1,
    y.d_values_,
    1));

  return handle_vector_operations.is_cuBLAS_success();
}

bool CuBLASVectorOperations::scalar_multiply(
  const float scalar,
  DenseVector& x,
  const std::size_t stride)
{
  HandleUnsuccessfulCuBLASCall handle_scalar_multiplication {
    "Failed to scalar multiply in cuBLAS"};

  const float alpha {scalar};

  // ref. https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublas-api
  // This function scales vector x by scalar alpha and overwrites it with the
  // result. Hence, performed operation is x[j] = \alpha * x[j]
  // Notice that i = 1...n, j = 1 + (i - 1) * incx reflect 1-based indexing used
  // for compatibility with Fortran.
  handle_scalar_multiplication(cublasSscal(
    cublas_handle_,
    x.number_of_elements_,
    &alpha,
    x.d_values_,
    static_cast<int>(stride)
    ));

  return handle_scalar_multiplication.is_cuBLAS_success();
}

bool CuBLASVectorOperations::scalar_multiply(
  const float scalar,
  Array& x,
  const std::size_t stride)
{
  HandleUnsuccessfulCuBLASCall handle_scalar_multiplication {
    "Failed to scalar multiply in cuBLAS"};

  const float alpha {scalar};

  // ref. https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublas-api
  // This function scales vector x by scalar alpha and overwrites it with the
  // result. Hence, performed operation is x[j] = \alpha * x[j]
  // Notice that i = 1...n, j = 1 + (i - 1) * incx reflect 1-based indexing used
  // for compatibility with Fortran.
  handle_scalar_multiplication(cublasSscal(
    cublas_handle_,
    x.number_of_elements_,
    &alpha,
    x.values_,
    static_cast<int>(stride)
    ));

  return handle_scalar_multiplication.is_cuBLAS_success();
}

optional<float> CuBLASVectorOperations::dot_product(
  const Array& r1,
  const Array& r2)
{
  HandleUnsuccessfulCuBLASCall handle_dot_product {
    "Failed dot product in cuBLAS"};

  float result {0.0f};

  handle_dot_product(cublasSdot(
    cublas_handle_,
    r1.number_of_elements_,
    r1.values_,
    1,
    r2.values_,
    1,
    &result));

  return handle_dot_product.is_cuBLAS_success() ? make_optional(result) :
    nullopt;
}

optional<float> CuBLASVectorOperations::dot_product(
  const DenseVector& r1,
  const DenseVector& r2)
{
  HandleUnsuccessfulCuBLASCall handle_dot_product {
    "Failed dot product in cuBLAS"};

  float result {0.0f};

  handle_dot_product(cublasSdot(
    cublas_handle_,
    r1.number_of_elements_,
    r1.d_values_,
    1,
    r2.d_values_,
    1,
    &result));

  return handle_dot_product.is_cuBLAS_success() ? make_optional(result) :
    nullopt;
}

bool CuBLASVectorOperations::copy(const Array& x, DenseVector& y)
{
  HandleUnsuccessfulCuBLASCall handle_copy {"Failed copy in cuBLAS"};

  handle_copy(cublasScopy(
    cublas_handle_,
    x.number_of_elements_,
    x.values_,
    1,
    y.d_values_,
    1));

  return handle_copy.is_cuBLAS_success();
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra
