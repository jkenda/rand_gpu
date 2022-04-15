#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>
#include <array>
#include <memory>

#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#include "../include/cl.hpp"

namespace rand_gpu
{
    class RNG_private
    {
    public:
        RNG_private(size_t multi);

        ~RNG_private();

        template <typename T>
        T get_random();

        size_t buffer_size();

        RNG_private(RNG_private&) = delete;
        RNG_private(RNG_private&&) = delete;
        RNG_private& operator=(RNG_private&) = delete;
        RNG_private& operator=(RNG_private&&) = delete;

    private:
        struct Buffer
        {
            std::vector<uint8_t> data;
            cl::Event ready_event;
            bool ready;
        };

        static void set_flag(cl_event e, cl_int status, void *data);

        std::mutex buffer_ready_lock;
        std::condition_variable buffer_ready_cond;

        std::unique_ptr<cl::CommandQueue> queue;

        cl::Buffer  state_buf;
        cl::Buffer  random_buf;
        cl::Kernel  k_generate;

        std::array<Buffer, 2> buffer;
        size_t buffer_max_size;
        uint_fast8_t active_buf;
        uint_fast8_t waiting_buf;

        size_t buf_offset;
    };

    size_t mem_usage();
}
