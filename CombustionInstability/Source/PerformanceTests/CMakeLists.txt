ADD_EXECUTABLE(BenchmarkCombustionInstability
  SpectralAnalysis/PSDPipeline_benchmark.cu
  )

TARGET_LINK_LIBRARIES(BenchmarkCombustionInstability
  SpectralAnalysisCU
  CUDA::cufft
  )

SET_TARGET_PROPERTIES(BenchmarkCombustionInstability
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

SET_TARGET_PROPERTIES(BenchmarkCombustionInstability
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/")
