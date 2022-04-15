/**
 * @file RNG.hpp
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

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#include "../include/cl.hpp"

namespace rand_gpu
{
    class RNG
    {
    public:
        /**
         * @brief Construct a new RNG object
         * 
         * @param multi buffer size multiplier
         */
        RNG(size_t multi);

        ~RNG();

        /**
         * @brief Returns next random number.
         * 
         * @tparam T type of random number to be returned - only \a uint<N>_t implemented
         */
        template <typename T = uint64_t>
        T get_random();

        /**
         * @brief Returns size of random number buffer in bytes
         * 
         * @return size_t 
         */
        size_t buffer_size();

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

        cl::CommandQueue queue;

        cl::Buffer  state_buf;
        cl::Buffer  random_buf;
        cl::Kernel  k_generate;

        std::array<Buffer, 2> buffer;
        size_t buffer_max_size;
        uint_fast8_t active_buf;
        uint_fast8_t waiting_buf;

        size_t buf_offset;
    };

} // namespace rand_gpu
