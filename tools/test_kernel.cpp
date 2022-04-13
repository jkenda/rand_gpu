#include <vector>
#include <cstdlib>
#include <iostream>
#include <cstring>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#include "../include/cl.hpp"
#include "../kernel.hpp"
#include "../src/exceptions.hpp"

using namespace std;

int main()
{
    cout << "Kernel syntax check..." << endl;

	// get platforms
    cl::Platform platform;
    int status = cl::Platform::get(&platform);
    if (status != CL_SUCCESS) throw RandGPUException("No OpenCL platforms found.");
    
    // get devices
    vector<cl::Device> devices;
    status = platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (status != CL_SUCCESS) throw RandGPUException("No OpenCL devices found.");

    // create devices
    cl::Device device = devices.at(0);

    cl::Context context = cl::Context(devices);
    cl::Program::Sources sources(1, make_pair(KERNEL_SOURCE, strlen(KERNEL_SOURCE)));
    cl::Program program = cl::Program(context, sources);

    program.build(devices, "");
    status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
    if (status == CL_BUILD_ERROR) {
        std::string name     = device.getInfo<CL_DEVICE_NAME>();
        std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        fputs(buildlog.c_str(), stderr);
        throw RandGPUException("Syntax check failed!");
    }
    cout << "Syntax check succeeded." << endl;
}
