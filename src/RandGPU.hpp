#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>

#define CL_TARGET_OPENCL_VERSION 120
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
    static RandGPU& instance(size_t multi);

    template <typename R>
    R rand();
    size_t buffer_size();

private:
    size_t nthreads;
    size_t wg_size;
    size_t buf_size;
    size_t buf_limit;

    cl::Context      context;
    cl::CommandQueue generate_queue;
    cl::CommandQueue cancel_queue;

	cl::Buffer       state_buf;
	cl::Buffer       random_buf;
	cl::Kernel       k_init;
	cl::Kernel       k_generate;

    Buffer buffer[2];

    uint_fast32_t active;
    uint_fast32_t buffer_i;

    RandGPU(size_t multi);
    RandGPU();
    RandGPU(const RandGPU&) = delete;
    RandGPU& operator= (const RandGPU&) = delete;
};
