// In main.cpp
#include "header.hpp"
#include <vector>
#include <iostream>
#include <exception> // Include for std::exception

int main() {
    std::cout << "THIS IS A SETUP EXAMPLE FOR CUDA SUPPORT,\nusing the CUDA Runtime API for Parallel Computing." << std::endl;
    std::vector<float> a = { 1.0f, 2.0f, 3.0f };
    std::vector<float> b = { 4.0f, 5.0f, 6.0f };
    std::vector<float> e = { 1.5f, 2.5f, 3.1f };
    std::vector<float> f = { 4.1f, 5.1f, 6.9f };
    std::vector<float> c(e.size());

    try {
        // Call the CUDA version
        addVectorsCUDA(a, b, c);
        std::cout << "Result from CUDA Kernel: ";
        for (float val : c) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        std::fill(c.begin(), c.end(), 0.0f);

        addVectorsCUDA(e, f, c); // Note: This function is still using the placeholder
        std::cout << "Result from Kernel file: ";
        for (float val : c) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        subtractVectorsCUDA(e, f, c);
        std::cout << "Result from Kernel file: ";
        for (float val : c) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    // Removed OpenCL specific catch block, CUDA errors are now runtime_error
    catch (const std::exception& e) { // Catch standard exceptions
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
