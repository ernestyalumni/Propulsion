#include "IntegrationTests/Visualization/GLUTInterface/JuliaSet/JuliaSet.h"

using IntegrationTests::Visualization::GLUTInterface::JuliaSet::JuliaSet;

int main(int argc, char* argv[])
{
  JuliaSet julia_set {};

  julia_set.run(&argc, argv);
}