
#include "header.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Kernel 1: Addition
__global__ void add_vectors(const float *a,
                            const float *b,
                            float *c,
                            const unsigned int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Kernel 2: Subtraction
__global__ void subtract_vectors(const float *a,
                                 const float *b,
                                 float *c,
                                 const unsigned int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}
