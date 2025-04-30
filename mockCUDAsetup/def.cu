
#include "header.hpp"
#include <fstream>
#include <stdexcept> // Ensure stdexcept is included for runtime_error
#include <cmath>     // For ceil

/**
 * @brief Adds vectors using a pre-compiled CUDA kernel.
 * @param a vector input for addition
 * @param b vector input for addition
 * @param res result to store vector addition
 * @note Assumes the 'add_vectors' CUDA kernel is compiled and linked.
 */
void addVectorsCUDA(std::vector<float>& a, std::vector<float>& b, std::vector<float>& res) {
    if (a.size() != b.size() || a.size() != res.size()) {
        throw std::runtime_error("Vector sizes do not match.");
    }

    unsigned int vector_size = static_cast<unsigned int>(a.size()); // Kernel expects unsigned int
    size_t vector_bytes = sizeof(float) * vector_size;

    // Initialize CUDA (selects device)
    CUDAContext cudaCtx; // Uses device 0 by default

    // Allocate memory on the CUDA device
    float *d_a = nullptr, *d_b = nullptr, *d_res = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, vector_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, vector_bytes));
    CUDA_CHECK(cudaMalloc(&d_res, vector_bytes));
    std::cout << "CUDA Memory Allocated." << std::endl;

    // Copy data from host (std::vector) to device (CUDA pointers)
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), vector_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), vector_bytes, cudaMemcpyHostToDevice));
    std::cout << "Data Copied Host -> Device." << std::endl;

    // Define kernel launch configuration (Grid and Block dimensions)
    int threadsPerBlock = 256;
    // Calculate grid size to cover all elements
    int blocksPerGrid = (vector_size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 gridDim(blocksPerGrid);
    dim3 blockDim(threadsPerBlock);

    // Launch the kernel
    add_vectors<<<gridDim, blockDim, 0, cudaCtx.stream>>>(d_a, d_b, d_res, vector_size);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
    std::cout << "CUDA Kernel Launched." << std::endl;

    // Copy results from device back to host
    CUDA_CHECK(cudaMemcpy(res.data(), d_res, vector_bytes, cudaMemcpyDeviceToHost));
    std::cout << "Data Copied Device -> Host." << std::endl;

    // Synchronize to ensure kernel and copies are finished (optional but good practice)
    // CUDA_CHECK(cudaStreamSynchronize(cudaCtx.stream)); // Sync specific stream
    CUDA_CHECK(cudaDeviceSynchronize()); // Sync entire device (simpler for this case)

    // Free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_res));
    std::cout << "CUDA Memory Freed." << std::endl;
}


/**
 * @brief Adds vectors using a pre-compiled CUDA kernel.
 * @param a vector input for addition
 * @param b vector input for addition
 * @param res result to store vector addition
 * @note Assumes the 'add_vectors' CUDA kernel is compiled and linked.
 */
void subtractVectorsCUDA(std::vector<float>& a, std::vector<float>& b, std::vector<float>& res) {
    if (a.size() != b.size() || a.size() != res.size()) {
        throw std::runtime_error("Vector sizes do not match.");
    }

    unsigned int vector_size = static_cast<unsigned int>(a.size()); // Kernel expects unsigned int
    size_t vector_bytes = sizeof(float) * vector_size;

    // Initialize CUDA (selects device)
    CUDAContext cudaCtx; // Uses device 0 by default

    // Allocate memory on the CUDA device
    float *d_a = nullptr, *d_b = nullptr, *d_res = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, vector_bytes));
    CUDA_CHECK(cudaMalloc(&d_b, vector_bytes));
    CUDA_CHECK(cudaMalloc(&d_res, vector_bytes));
    std::cout << "CUDA Memory Allocated." << std::endl;

    // Copy data from host (std::vector) to device (CUDA pointers)
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), vector_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), vector_bytes, cudaMemcpyHostToDevice));
    std::cout << "Data Copied Host -> Device." << std::endl;

    // Define kernel launch configuration (Grid and Block dimensions)
    int threadsPerBlock = 256;
    // Calculate grid size to cover all elements
    int blocksPerGrid = (vector_size + threadsPerBlock - 1) / threadsPerBlock;
    dim3 gridDim(blocksPerGrid);
    dim3 blockDim(threadsPerBlock);

    // Launch the kernel
    subtract_vectors<<<gridDim, blockDim, 0, cudaCtx.stream>>>(d_a, d_b, d_res, vector_size);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
    std::cout << "CUDA Kernel Launched." << std::endl;

    // Copy results from device back to host
    CUDA_CHECK(cudaMemcpy(res.data(), d_res, vector_bytes, cudaMemcpyDeviceToHost));
    std::cout << "Data Copied Device -> Host." << std::endl;

    // Synchronize to ensure kernel and copies are finished (optional but good practice)
    // CUDA_CHECK(cudaStreamSynchronize(cudaCtx.stream)); // Sync specific stream
    CUDA_CHECK(cudaDeviceSynchronize()); // Sync entire device (simpler for this case)

    // Free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_res));
    std::cout << "CUDA Memory Freed." << std::endl;
}
