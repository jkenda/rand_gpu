#include <iostream>
#include <string>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#include "../src/include/cl.hpp"
#include "../kernel.hpp"

using namespace std;

int main()
{
    cout << "Kernel syntax check..." << endl;

	// get platforms
    cl::Platform platform;
    cl::Platform::get(&platform);
    
    // get devices
    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    // create devices
    cl::Device device = devices.at(0);
    size_t len = 0;

    cl::Context context = cl::Context(devices);
    cl::Program::Sources sources;
    for (int i = 0; i < 15; i++) {
        sources.emplace_back(KERNEL_SOURCE[i].c_str(), KERNEL_SOURCE[i].length());
        len += KERNEL_SOURCE[i].length();
    }
    cl::Program program = cl::Program(context, sources);

    try
    {
        program.build(devices, "");
        cout << "size: " << len << '\n';
        cout << "Syntax check succeeded." << '\n';
    }
    catch (const cl::Error& err)
    {
        std::string name     = device.getInfo<CL_DEVICE_NAME>();
        std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        cerr << buildlog << '\n';
        cout << "size: " << len << " bytes\n";
        cerr << "Syntax check failed!" << '\n';
        return 13;
    }
}
