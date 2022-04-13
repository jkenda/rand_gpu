#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#include "../include/cl.hpp"


struct Buffer
{
    std::vector<uint8_t> data;
    cl::Event ready_event;
    bool ready;
};

class RandGPU
{
public:
    RandGPU(size_t multi);

    template <typename R>
    R rand();
    size_t buffer_size();

private:
    cl::CommandQueue queue;

	cl::Buffer  random_buf;
	cl::Kernel  k_generate;

    Buffer buffer[2];

    size_t active;
    size_t buffer_i;
};
