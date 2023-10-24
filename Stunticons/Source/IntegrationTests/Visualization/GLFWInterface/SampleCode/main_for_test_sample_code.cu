//#include "DataStructures/Array.h"
#include "IntegrationTests/Visualization/GLFWInterface/SampleCode/SampleCode.h"
//#include "IntegrationTests/Visualization/GLFWInterface/SampleCode/fill_RGB.h"
//#include "Visualization/GLFWInterface/GLFWWindow.h"

/*
#include <GLFW/glfw3.h>
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
*/

/*
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
*/

//using DataStructures::Array;
using IntegrationTests::Visualization::GLFWInterface::SampleCode::SampleCode;
//using IntegrationTests::Visualization::GLFWInterface::SampleCode::fill_RGB;
//using Visualization::GLFWInterface::GLFWWindow;
//using std::size_t;

// Original code
// CUDA kernel to fill the RGB buffer with values
/*
__global__ void fillRGB(unsigned char *rgb) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = idx * 3;
    rgb[offset] = idx % 255;     // R value
    rgb[offset + 1] = (idx * 3) % 255; // G value
    rgb[offset + 2] = (idx * 7) % 255; // B value
}
*/

int main(int argc, char* argv[])
{
  SampleCode sample_code {};

  sample_code.run_sample_code(&argc, argv);

  // This works.
  /*
  GLFWWindow glfw_window {};
  glfw_window.initialize();
  glfw_window.create_window(SampleCode::default_glfw_window_parameters_);
  */

  // This was the original code that is replaced by the above GLFWWindow object
  // and calls.
  /*
  GLFWwindow* window;

  // Initialize GLFW
  if (!glfwInit()) {
      fprintf(stderr, "Failed to initialize GLFW\n");
      return -1;
  }

  // Create a windowed mode window and its OpenGL context
  window = glfwCreateWindow(SampleCode::width_, SampleCode::height_, "Custom RGB Window", NULL, NULL);
  if (!window) {
      fprintf(stderr, "Failed to create GLFW window\n");
      glfwTerminate();
      return -1;
  }

  // Make the window's context current
  glfwMakeContextCurrent(window);
  */

  // This works.
  /*
  Array<unsigned char> array {
    SampleCode::default_glfw_window_parameters_.width_ *
      SampleCode::default_glfw_window_parameters_.height_ *
      3};
  */

  // This was the original implementation of the memory allocation.
  /*
  // Allocate memory for RGB buffer on device
  unsigned char *dev_rgb;
  cudaMalloc((void**)&dev_rgb, SampleCode::width_ * SampleCode::height_ * 3 * sizeof(unsigned char));
  */

  /*
  // Fill RGB buffer with values using kernel
  const size_t threadsPerBlock {256};
  const size_t blocksPerGrid {(SampleCode::width_ * SampleCode::height_ + threadsPerBlock - 1) / threadsPerBlock};
  */
  //fill_RGB<<<blocksPerGrid, threadsPerBlock>>>(dev_rgb);

  /*
  fill_RGB<<<blocksPerGrid, threadsPerBlock>>>(array.elements_);

  // Allocate memory for RGB buffer on host
  unsigned char *host_rgb {
    new unsigned char[SampleCode::width_ * SampleCode::height_ * 3]};

  // Copy RGB buffer from device to host
  //cudaMemcpy(host_rgb, dev_rgb, SampleCode::width_ * SampleCode::height_ * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  // cudaMemcpy(host_rgb, array.elements_, SampleCode::width_ * SampleCode::height_ * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  array.copy_device_output_to_host(host_rgb);

  // Loop until the user closes the window
  /*
  while (!glfwWindowShouldClose(window)) {
      // Render RGB buffer to window
      glClear(GL_COLOR_BUFFER_BIT);
      glDrawPixels(SampleCode::width_, SampleCode::height_, GL_RGB, GL_UNSIGNED_BYTE, host_rgb);
      glfwSwapBuffers(window);
      glfwPollEvents();
  }

  // Clean up GLFW
  glfwTerminate();
  */

  // This works.
  /*
  while (!glfwWindowShouldClose(glfw_window.created_window_handle_))
  {
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(SampleCode::width_, SampleCode::height_, GL_RGB, GL_UNSIGNED_BYTE, host_rgb);
    glfwSwapBuffers(glfw_window.created_window_handle_);
    glfwPollEvents();    
  }

  // Clean up memory
  // cudaFree(dev_rgb);
  delete[] host_rgb;
  */

  return 0;
}