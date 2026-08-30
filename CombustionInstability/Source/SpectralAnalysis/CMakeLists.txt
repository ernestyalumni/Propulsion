SET(SPECTRAL_ANALYSIS_CU_SOURCES
  # Header-only kernels: no .cu sources needed here.
  # cuFFT wrapper functions live in the consuming executables.
  )

# SpectralAnalysisCU is a header-only interface library so that unit tests
# and performance tests can link against cuFFT without duplicating flags.
ADD_LIBRARY(SpectralAnalysisCU INTERFACE)

TARGET_INCLUDE_DIRECTORIES(SpectralAnalysisCU
  INTERFACE ${CMAKE_SOURCE_DIR})

TARGET_LINK_LIBRARIES(SpectralAnalysisCU
  INTERFACE CUDA::cufft)
