/**
 * @file RNG.hpp
 * @author Jakob Kenda (kenda.jakob@domain.com)
 * @brief 
 * @version 0.3
 * @date 2022-04-13
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <chrono>
#include "rand_gpu.h"

// forward declaration of the hidden struct
struct RNG_private;


namespace rand_gpu
{
    template <rand_gpu_algorithm A = RAND_GPU_ALGORITHM_TYCHE>
    class RNG
    {
    public:
        /**
         * @brief Construct a new RNG object
         * 
         */
        RNG();

        /**
         * @brief Construct a new RNG object
         * 
         * @param algorithm Algorithm for the RNG
         * @param n_buffers Number of buffers for storing random numbers
         * @param multi buffer size multiplier
         */
        RNG(size_t n_buffers, size_t multi);

        /**
         * @brief Construct a new RNG object
         * 
         * @param seed Custom seed
         * @param algorithm Algorithm for the RNG
         * @param n_buffers Number of buffers for storing random numbers
         * @param multi buffer size multiplier
         */
        RNG(uint64_t seed, size_t n_buffers, size_t multi);

        /**
         * @brief Destroy the RNG object
         * 
         */
        ~RNG();

        /**
         * @brief Returns next random number.
         * @tparam T type of random number to be returned - implemented for all primitive types, 
         *         however, only 32- and 64-bit numbers have been tested for true randomness. 
         */
        template <typename T = uint32_t>
        T get_random();

        /**
         * @brief Returns size of random number buffer in bytes
         */
        std::size_t buffer_size() const;

        /**
         * @brief Returns number of times we had to wait for GPU to fill the buffer.
         */
        std::size_t buffer_misses() const;

        /**
         * @brief Returns RNG initialization time.
         */
        std::chrono::nanoseconds init_time() const;

        /**
         * @brief Return average GPU transfer time.
         */
        std::chrono::nanoseconds avg_gpu_transfer_time() const;

        // deleted copy constructor and assignment operator
        RNG(RNG&) = delete;
        RNG& operator=(RNG&) = delete;
        RNG(RNG&&);
        RNG& operator=(RNG&&);

    private:
        RNG_private *d_ptr_;
    };

    /**
     * @brief Returns memory used by all RNG instances.
     */
    std::size_t memory_usage();

    /**
     * @brief Returns number of times we had to wait for GPU to fill the buffer.
     */
    std::size_t buffer_misses();

    /**
     * @brief Return average init time of all RNGs.
     */
    std::chrono::nanoseconds avg_init_time();

    /**
     * @brief Return average GPU transfer time of all RNGs.
     */
    std::chrono::nanoseconds avg_gpu_transfer_time();

    /**
     * @brief Returns the name of the algorithm corresponding to the enum.
     * 
     * @param algorithm enum of the algorithm
     * @param long_name bool the name along with description
     */
    const char *algorithm_name(rand_gpu_algorithm algorithm, bool long_name = false);
}
