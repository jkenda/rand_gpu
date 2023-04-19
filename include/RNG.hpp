/**
 * @file RNG.hpp
 * @author Jakob Kenda (kenda.jakob@gmail.com)
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
struct RNG_impl;


namespace rand_gpu
{
    template <rand_gpu_algorithm A = RAND_GPU_ALGORITHM_TYCHE_I>
    class RNG
    {
    public:
        /**
         * @brief Construct a new RNG object
         */
        RNG();

        /**
         * @brief Construct a new RNG object that wraps rng.
         *
         * @param rng rng to wrap the newly constructed RNG around
         */
        RNG(rand_gpu_rng rng);

        /**
         * @brief Construct a new RNG object
         * 
         * @param n_buffers Number of buffers for storing random numbers
         * @param multi Buffer size multiplier
         */
        RNG(size_t n_buffers, size_t multi);

        /**
         * @brief Construct a new RNG object
         * 
         * @param seed Custom seed
         * @param n_buffers Number of buffers for storing random numbers
         * @param multi Buffer size multiplier
         */
        RNG(size_t n_buffers, size_t multi, uint64_t seed);

        /**
         * @brief Destroy the RNG object
         */
        ~RNG();

        /**
         * @brief Returns next random number.
         * @tparam T type of random number to be returned - implemented for all primitive types.
         */
        template <typename T>
        T get_random();

        /**
         * @brief Returns next random number.
         * @param dst where to put the random bytes
         * @param nbytes how many bytes to copy
         */
        void put_random(void *dst, const size_t nbytes);

        /**
         * @brief Returns next random number.
         */
        uint64_t operator()();

        /**
         * @brief Discards z bytes from buffer.
         */
        void discard(size_t z);

        /**
         * @brief Returns size of random number buffer in bytes
         */
        std::size_t buffer_size() const;

        /**
         * @brief Returns size of RNG state in bytes
         */
        std::size_t state_size() const;

        /**
         * @brief Returns number of times the buffer was switched.
         */
        std::size_t buffer_switches() const;

        /**
         * @brief Returns number of times we had to wait for GPU to fill the buffer.
         */
        std::size_t buffer_misses() const;

        /**
         * @brief Returns RNG initialization time.
         */
        std::chrono::nanoseconds init_time() const;

        /**
         * @brief Returns average GPU transfer time.
         */
        std::chrono::nanoseconds avg_gpu_calculation_time() const;

        /**
         * @brief Returns average transfer time for GPU in (including time spent waiting for calculations).
         */
        std::chrono::nanoseconds avg_gpu_transfer_time() const;

        // deleted copy constructor and assignment operator
        RNG(RNG&) = delete;
        RNG& operator=(RNG&) = delete;
        RNG(RNG&&);
        RNG& operator=(RNG&&);

    private:
        RNG_impl *_impl_ptr;
    };

    /**
     * @brief Returns memory used by all RNG instances.
     */
    std::size_t memory_usage();

    /**
     * @brief Returns number of times the buffer was switched in all RNG instances.
     */
    std::size_t buffer_switches();

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
    std::chrono::nanoseconds avg_gpu_calculation_time();

    /**
     * @brief Return average GPU transfer time of all RNGs (including time spent waiting for calculations).
     */
    std::chrono::nanoseconds avg_gpu_transfer_time();

    /**
     * @brief Returns the compilation time for the algorithm.
     * 
     * @param algorithm enum of the algorithm
     */
    std::chrono::nanoseconds compilation_time(rand_gpu_algorithm algorithm);

    /**
     * @brief Returns the name of the algorithm corresponding to the enum.
     * 
     * @param algorithm enum of the algorithm
     * @param long_name bool the name along with description
     */
    const char *algorithm_name(rand_gpu_algorithm algorithm, bool long_name = false);
}
