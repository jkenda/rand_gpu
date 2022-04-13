/**
 * @file RandGPU.hpp
 * @author Jakob Kenda (kenda.jakob@domain.com)
 * @brief 
 * @version 0.2
 * @date 2022-04-13
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>
#include <array>
#include <memory>

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

    // copy/move
    RandGPU(RandGPU&&);
    RandGPU(const RandGPU&) = delete;
    RandGPU& operator=(RandGPU&&) = default;
    RandGPU& operator=(const RandGPU&) = delete;

private:
    std::unique_ptr<std::mutex> buffer_ready_lock;
    std::unique_ptr<std::condition_variable> buffer_ready_cond;

    cl::CommandQueue queue;

    cl::Buffer  state_buf;
	cl::Buffer  random_buf;
	cl::Kernel  k_generate;

    std::array<Buffer, 2> buffer;

    size_t active;
    size_t buffer_i;

    static void set_flag(cl_event e, cl_int status, void *data);
};
