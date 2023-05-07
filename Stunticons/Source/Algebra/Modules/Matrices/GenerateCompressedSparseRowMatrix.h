#ifndef ALGEBRA_MODULES_MATRICES_GENERATE_COMPRESSED_SPARSE_ROW_MATRIX_H
#define ALGEBRA_MODULES_MATRICES_GENERATE_COMPRESSED_SPARSE_ROW_MATRIX_H

#include "HostCompressedSparseRow.h"

namespace Algebra
{
namespace Modules
{
namespace Matrices
{
namespace CompressedSparseRow
{

//------------------------------------------------------------------------------
/// \ref https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/conjugateGradient
//------------------------------------------------------------------------------
void generate_tridiagonal_matrix(HostCompressedSparseRowMatrix& a)

} // namespace CompressedSparseRow

} // namespace Matrices
} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_MATRICES_GENERATE_COMPRESSED_SPARSE_ROW_MATRIX_H