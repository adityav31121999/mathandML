#ifndef HEADER_HPP
#define HEADER_HPP 1

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <map>
#include <cuda_runtime.h> // Use cuda_runtime.h for Runtime API

// --- CUDA Helper Macro for Error Checking ---
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            throw std::runtime_error(cudaGetErrorString(err));      \
        }                                                           \
    } while (0)

// --- Basic CUDA Context Helper Class ---
class CUDAContext {
public:
    int deviceId;           // ID of the selected CUDA device
    cudaStream_t stream;    // CUDA stream (optional, can use default 0)
    cudaDeviceProp deviceProp; // Properties of the selected device

    // Constructor: Selects device 0 by default and creates a stream
    CUDAContext(int devId = 0) : deviceId(devId), stream(0) {
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            throw std::runtime_error("No CUDA-enabled devices found.");
        }
        if (devId >= deviceCount) {
             throw std::runtime_error("Selected device ID is invalid.");
        }
        deviceId = devId;
        CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, deviceId));
        // Optionally create a non-default stream
        // CUDA_CHECK(cudaStreamCreate(&stream));
        std::cout << "CUDA Context Initialized on Device " << deviceId << ": " << deviceProp.name << std::endl;
    }

    // Destructor: Cleans up resources (e.g., the stream if created)
    ~CUDAContext() {
        // If a non-default stream was created, destroy it
        // if (stream != 0) {
        //     cudaError_t err = cudaStreamDestroy(stream);
        //     // Don't throw from destructor, just report
        //     if (err != cudaSuccess) {
        //         fprintf(stderr, "CUDA Error during stream destruction: %s\n", cudaGetErrorString(err));
        //     }
        // }
        // cudaDeviceReset(); // Usually not needed here, resets the *entire* device context
    }

    // Disable copy and assignment
    CUDAContext(const CUDAContext&) = delete;
    CUDAContext& operator=(const CUDAContext&) = delete;
};

extern __global__ void add_vectors(const float *a, const float *b, float *c, const unsigned int n);
extern __global__ void subtract_vectors(const float *a, const float *b, float *c, const unsigned int n);

// Function declarations (match the functions currently defined in def.cpp)
void addVectorsCUDA(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c);
void subtractVectorsCUDA(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c);

#endif // HEADER_HPP
