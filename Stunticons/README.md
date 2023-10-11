# Setting up CUDA and OpenGL Interoperability

If you are having difficulty when running, for instance, the [CUDA samples](https://github.com/NVIDIA/cuda-samples) that uses OpenGL graphics, such as the following error:

```
CUDA error at fluidsGL.cpp:467 code=999(cudaErrorUnknown) "cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone)" 
```

Then first try to diagnose which GPU is available and being used:

```
lspci | grep VGA
```

You could try to use `prime-select` to use NVIDIA GPU exclusively, although, for example if there was no integrated GPU, then `prime-select query` resulting in `on-demand` would still allow for the samples to run fine:

```
sudo prime-select nvidia
```

https://forums.developer.nvidia.com/t/cuda-11-6-opengl-interoperability-broken/214581/3

https://forums.developer.nvidia.com/t/cuda-11-6-opengl-interoperability-broken/214581/7?u=ernestyalumni

*tl;dr*:


PROBLEM: OpenGL context was being instantiated on integrated graphics (Intel) GPU, causing CUDA-GL interoperability to fail.

SOLUTION: As per https://wiki.debian.org/NVIDIA%20Optimus 39, set a couple of environment variables before running the executable (in this case the executable is “./volumeRender”). This makes OpenGL run on the NVIDIA card, so that it can interface properly with CUDA:

```
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./volumeRender
```

## Installing necessary packages

Given

https://developer.download.nvidia.com/compute/cuda/6_5/rel/docs/CUDA_Getting_Started_Linux.pdf

I ran this in Ubuntu Linux:

```
sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev
 libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev
```
and
```

```