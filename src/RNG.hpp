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

#include <memory>

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
         * @param multi buffer size multiplier
         */
        RNG(size_t multi = 2);

        /**
         * @brief Destroy the RNG object
         * 
         */
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

    /**
     * @brief Returns memory used by all RNG instances.
     * 
     * @return size_t memory used by all RNG instances
     */
    size_t memory_usage();
}
