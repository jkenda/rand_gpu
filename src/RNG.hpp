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
#include "rand_gpu.h"

// forward declaration of the private class
class RNG_private;


namespace rand_gpu
{
    class RNG
    {
    public:
        /**
         * @brief Construct a new RNG object
         * 
         * @param algorithm Algorithm for the RNG
         * @param n_buffers Number of buffers for storing random numbers
         * @param multi buffer size multiplier
         */
        RNG(rand_gpu_algorithm algorithm = RAND_GPU_ALGORITHM_TYCHE, size_t n_buffers = 2, size_t multi = 1);

        /**
         * @brief Construct a new RNG object
         * 
         * @param seed Custom seed
         * @param algorithm Algorithm for the RNG
         * @param n_buffers Number of buffers for storing random numbers
         * @param multi buffer size multiplier
         */
        RNG(uint64_t seed, rand_gpu_algorithm algorithm = RAND_GPU_ALGORITHM_TYCHE, size_t n_buffers = 2, size_t multi = 1);

        /**
         * @brief Destroy the RNG object
         * 
         */
        ~RNG();

        /**
         * @brief Returns next random number.
         *        DO NOT MIX TESTED AND UNTESTED TYPES!
         * 
         * @tparam T type of random number to be returned - implemented for all primitive types, 
         *         however, only 32- and 64-bit numbers have been tested for true randomness. 
         */
        template <typename T = unsigned long>
        T get_random();

        /**
         * @brief Returns size of random number buffer in bytes
         * 
         * @return size_t size of random number buffer in bytes
         */
        size_t buffer_size() const;

        /**
         * @brief Returns number of times we had to wait for GPU to fill the buffer.
         * 
         * @return size_t number of buffer misses
         */
        size_t buffer_misses() const;

        /**
         * @brief Returns RNG initialization time in ms
         * 
         * @return float RNG initialization time in ms
         */
        float init_time() const;

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
     * 
     * @return size_t memory used by all RNG instances
     */
    size_t memory_usage();

    const char *algorithm_name(rand_gpu_algorithm algorithm, bool long_name = false);
}
