#include "header.hpp"
#include <fstream>
#include <stdexcept> // Ensure stdexcept is included for runtime_error

// raw string for kernel source
const char* kernelSourceCode = R"(
    __kernel void add_vectors(__global const float *a, __global const float *b, __global float *c, const uint n) {
        int i = get_global_id(0);
        if (i < n) {
            c[i] = a[i] + b[i];
        }
    }
)";

/**
 * @brief add vectors via kernel from string literal
 * @param a vector input for addition
 * @param b vector input for addition
 * @param res result to store vector addition
 * @note to create opencl context, OpenCLContext class
 *      is used with string to hold kernel source from
 *      string literal with kernal name
 */
void addVectorWithKernelString(std::vector<float>& a, std::vector<float>& b, std::vector<float>& res) {
    if (a.size() != b.size() || a.size() != res.size()) {
        throw std::runtime_error("Vector sizes do not match.");
    }
    size_t vector_size = a.size(); // Use a variable for clarity
    size_t vector_bytes = sizeof(float) * vector_size; // Calculate byte size correctly
    
    std::string kernelSource(kernelSourceCode);
    
    // Create OpenCL context (use name of kernel, not string literal)
    OpenCLContext clContext(kernelSource, "add_vectors");
    std::cout << "String Read." << std::endl;
    // Create buffers using the correct size (sizeof(float))
    cl::Buffer bufferA(clContext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vector_bytes, a.data());
    cl::Buffer bufferB(clContext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vector_bytes, b.data());
    cl::Buffer bufferRes(clContext.context, CL_MEM_WRITE_ONLY, vector_bytes); // Allocate only

    // Set kernel arguments
    clContext.kernel.setArg(0, bufferA);
    clContext.kernel.setArg(1, bufferB);
    clContext.kernel.setArg(2, bufferRes);
    // Pass size_t directly if kernel expects it, or cast safely if it expects int/uint
    // Assuming the kernel expects an integer type representing the number of elements
    clContext.kernel.setArg(3, static_cast<cl_uint>(vector_size)); // Use cl_uint for OpenCL size/index types
    std::cout << "Buffers Created." << std::endl;
    // Execute kernel
    cl::NDRange globalSize(vector_size);
    // Consider adding localSize if needed for performance, otherwise NullRange is fine
    clContext.queue.enqueueNDRangeKernel(clContext.kernel, cl::NullRange, globalSize);

    // Read results using the correct size (sizeof(float))
    clContext.queue.enqueueReadBuffer(bufferRes, CL_TRUE, 0, vector_bytes, res.data());

    // It's good practice to finish the queue to ensure completion and catch errors
    clContext.queue.finish();
    std::cout << "Kernel Executed." << std::endl;
}

/**
 * @brief add vectors via kernel from kernel (.cl) file
 * @param a vector input for addition
 * @param b vector input for addition
 * @param res result to store vector addition
 * @note to create opencl context use OpenCLContext class 
 *      with string to hold kernel file with name of kernel
 *      as second argument
 */
void addVectorWithKernelFile(std::vector<float>& a, std::vector<float>& b, std::vector<float>& res) {
    if (a.size() != b.size() || a.size() != res.size()) {
        throw std::runtime_error("Vector sizes do not match.");
    }
    size_t vector_size = a.size(); // Use a variable for clarity
    size_t vector_bytes = sizeof(float) * vector_size; // Calculate byte size correctly

    // Read kernel source from file
    std::ifstream kernelFile("D:/mockCLsetup/kernel.cl");
    if (!kernelFile.is_open()) {
        // Provide more context in the error message
        throw std::runtime_error("Could not open kernel file: D:/mockCLsetup/kernel.cl");
    }
    std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
    kernelFile.close();
    std::cout << "File Read." << std::endl;
    // Create OpenCL context (potential exceptions here)
    OpenCLContext clContext(kernelSource, "add_vectors"); 

    // Create buffers using the correct size (sizeof(float))
    cl::Buffer bufferA(clContext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vector_bytes, a.data());
    cl::Buffer bufferB(clContext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vector_bytes, b.data());
    cl::Buffer bufferRes(clContext.context, CL_MEM_WRITE_ONLY, vector_bytes); // Allocate only
    std::cout << "Buffers Created." << std::endl;
    // Set kernel arguments
    clContext.kernel.setArg(0, bufferA);
    clContext.kernel.setArg(1, bufferB);
    clContext.kernel.setArg(2, bufferRes);
    // Pass size_t directly if kernel expects it, or cast safely if it expects int/uint
    // Assuming the kernel expects an integer type representing the number of elements
    clContext.kernel.setArg(3, static_cast<cl_uint>(vector_size)); // Use cl_uint for OpenCL size/index types

    // Execute kernel
    cl::NDRange globalSize(vector_size);
    // Consider adding localSize if needed for performance, otherwise NullRange is fine
    clContext.queue.enqueueNDRangeKernel(clContext.kernel, cl::NullRange, globalSize);

    // Read results using the correct size (sizeof(float))
    clContext.queue.enqueueReadBuffer(bufferRes, CL_TRUE, 0, vector_bytes, res.data());

    // It's good practice to finish the queue to ensure completion and catch errors
    clContext.queue.finish();
}


/**
 * @brief subtract vectors via kernel from kernel (.cl) file
 * @param a vector input for addition
 * @param b vector input for addition
 * @param res result to store vector subtraction
 * @note to create opencl context use OpenCLContext class 
 *      with string to hold kernel file with name of kernel
 *      as second argument
 */
void subtractVectorWithKernelFile(std::vector<float>& a, std::vector<float>& b, std::vector<float>& res) {
    if (a.size() != b.size() || a.size() != res.size()) {
        throw std::runtime_error("Vector sizes do not match.");
    }
    size_t vector_size = a.size(); // Use a variable for clarity
    size_t vector_bytes = sizeof(float) * vector_size; // Calculate byte size correctly

    // Read kernel source from file
    std::ifstream kernelFile("D:/mockCLsetup/kernel.cl");
    if (!kernelFile.is_open()) {
        // Provide more context in the error message
        throw std::runtime_error("Could not open kernel file: D:/mockCLsetup/kernel.cl");
    }
    std::string kernelSource((std::istreambuf_iterator<char>(kernelFile)), std::istreambuf_iterator<char>());
    kernelFile.close();
    std::cout << "File Read." << std::endl;
    // Create OpenCL context (potential exceptions here)
    OpenCLContext clContext(kernelSource, "subtract_vectors");

    // Create buffers using the correct size (sizeof(float))
    cl::Buffer bufferA(clContext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vector_bytes, a.data());
    cl::Buffer bufferB(clContext.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vector_bytes, b.data());
    cl::Buffer bufferRes(clContext.context, CL_MEM_WRITE_ONLY, vector_bytes); // Allocate only
    std::cout << "Buffers Created." << std::endl;
    // Set kernel arguments
    clContext.kernel.setArg(0, bufferA);
    clContext.kernel.setArg(1, bufferB);
    clContext.kernel.setArg(2, bufferRes);
    // Pass size_t directly if kernel expects it, or cast safely if it expects int/uint
    // Assuming the kernel expects an integer type representing the number of elements
    clContext.kernel.setArg(3, static_cast<cl_uint>(vector_size)); // Use cl_uint for OpenCL size/index types

    // Execute kernel
    cl::NDRange globalSize(vector_size);
    // Consider adding localSize if needed for performance, otherwise NullRange is fine
    clContext.queue.enqueueNDRangeKernel(clContext.kernel, cl::NullRange, globalSize);

    // Read results using the correct size (sizeof(float))
    clContext.queue.enqueueReadBuffer(bufferRes, CL_TRUE, 0, vector_bytes, res.data());

    // It's good practice to finish the queue to ensure completion and catch errors
    clContext.queue.finish();
}
