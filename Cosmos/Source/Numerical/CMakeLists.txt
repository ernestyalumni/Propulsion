ADD_SUBDIRECTORY(ODE/RKMethods)

ADD_LIBRARY(Numerical
  ExponentiationBySquaring.cpp
  ODE/Output.cpp
  ODE/RK4.cpp
  ODE/RKMethods/Coefficients/DOPR853Coefficients.cpp
  ODE/RKMethods/Coefficients/RK4Coefficients.cpp
  ODE/RhsVan.cpp
  Optimization/mins.cpp
  )
TARGET_LINK_LIBRARIES(Numerical PUBLIC CosmosODE)
