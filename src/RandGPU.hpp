#pragma once

#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>

#define CL_TARGET_OPENCL_VERSION 120
#include "../include/cl.hpp"


struct Buffer
{
    std::vector<uint8_t> data;
    bool ready;
};

class RandGPU
{
public:
    static RandGPU &instance(size_t multi);

    template <typename R>
    R rand();
    size_t buffer_size();

private:
    size_t nthreads;
    size_t buf_size;
    size_t buf_limit;

    cl::Context      context;
    cl::CommandQueue queue;
	cl::Buffer       state_buf;
	cl::Buffer       random_buf;
	cl::Kernel       k_init;
	cl::Kernel       k_generate;

    cl::NDRange global_range;
    cl::NDRange local_range;

    Buffer buffer[2];

    uint_fast32_t active_buffer = 0;
    uint_fast32_t buffer_i = 1;

    cl::Event buffer_ready_event;

    RandGPU(size_t multi);
    RandGPU();
    RandGPU(const RandGPU&) = delete;
    RandGPU& operator= (const RandGPU&) = delete;
};
