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

#include <memory>

namespace rand_gpu
{
    class RNG_private;

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
        template <typename T = unsigned long>
        T get_random();

        /**
         * @brief Returns size of random number buffer in bytes
         * 
         * @return size_t 
         */
        size_t buffer_size();

        // deleted move and copy constructors and assignment operators
        RNG(RNG&) = delete;
        RNG(RNG&&) = delete;
        RNG& operator=(RNG&) = delete;
        RNG& operator=(RNG&&) = delete;

    private:
        std::unique_ptr<RNG_private> d_ptr_;
    };

    size_t memory_usage();
}
