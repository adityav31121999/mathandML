#ifndef HEADER_HPP
#define HEADER_HPP 1

#define CL_HPP_ENABLE_EXCEPTIONS // Use exceptions for error handling
#define CL_HPP_TARGET_OPENCL_VERSION 300 // Specify target version

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <CL/cl.hpp>

class OpenCLContext {
public:
    cl::Context context;        // Represents an OpenCL context, which manages devices and memory.
    cl::Device device;          // Represents an OpenCL device (e.g., GPU or CPU).
    cl::CommandQueue queue;     // Represents a command queue, used to submit commands to a device.
    cl::Program program;        // Represents an OpenCL program, which contains the compiled kernel code.
    cl::Kernel kernel;          // Represents an OpenCL kernel, which is a function that runs on the device.

    OpenCLContext(const std::string& kernelSource, const std::string& kernelName) {
        try {
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            if (platforms.empty()) {
                throw std::runtime_error("No OpenCL platforms found.");
            }

            // --- Platform Info ---
            cl::Platform platform = platforms[0]; // Use the first platform

            // --- Device Selection (Prefer GPU, fallback to CPU) ---
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices); // Try GPU first
            if (devices.empty()) {
                std::cerr << "Warning: No OpenCL GPU devices found. Trying CPU..." << std::endl;
                platform.getDevices(CL_DEVICE_TYPE_CPU, &devices); // Fallback to CPU
                if (devices.empty()) {
                   throw std::runtime_error("No OpenCL devices found (GPU or CPU).");
                }
            }
            device = devices[0]; // Select the first available device
            // --- Context & Queue ---
            context = cl::Context(device);
            queue = cl::CommandQueue(context, device);

            // --- Program Build ---
            cl::Program::Sources sources;
            sources.push_back({ kernelSource.c_str(), kernelSource.length() });
            program = cl::Program(context, sources);

            try {
                program.build({ device });
            } 
            catch (cl::BuildError &e) {
                throw std::runtime_error("Error building OpenCL program: " + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
            }

            kernel = cl::Kernel(program, kernelName.c_str());
        }
        catch (cl::Error &err) {
            throw std::runtime_error("OpenCL Error during setup: " + std::string(err.what()) + " (" + std::to_string(err.err()) + ")");
        }
    }

    ~OpenCLContext() = default;
};

// Function declarations (match the functions currently defined in def.cpp)
void addVectorWithKernelString(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c);
void addVectorWithKernelFile(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c);
void subtractVectorWithKernelFile(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c);

#endif // HEADER_HPP
