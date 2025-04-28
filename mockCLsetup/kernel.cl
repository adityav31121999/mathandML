// Kernel 1: Addition
__kernel void add_vectors(__global const float *a,
                          __global const float *b,
                          __global float *c,
                          const uint n)
{
    int i = get_global_id(0);
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Kernel 2: Subtraction
__kernel void subtract_vectors(__global const float *a,
                               __global const float *b,
                               __global float *c,
                               const uint n)
{
    int i = get_global_id(0);
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}

// You could also add helper functions (without __kernel) here
// float my_helper_function(float x) { return x * x; }
