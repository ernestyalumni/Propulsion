ADD_LIBRARY(Testing
  Manifolds/Operators/FiniteDifference/2D/DirectionalDerivativesKernel.cu
  Manifolds/Operators/FiniteDifference/2D/Stencil.cu
  )

TARGET_LINK_LIBRARIES(Testing
  MANIFOLDS
  )