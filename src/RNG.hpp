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
         * @param seed Custom seed
         * @param multi buffer size multiplier
         */
        RNG(const size_t multi = 1, const uint64_t seed = 0);

        /**
         * @brief Destroy the RNG object
         * 
         */
        ~RNG();

        /**
         * @brief Returns next random number.
         * 
         * @tparam T type of random number to be returned - implemented for all primitive types
         */
        template <typename T = unsigned long>
        T get_random();

        /**
         * @brief Returns size of random number buffer in bytes
         * 
         * @return size_t 
         */
        const size_t buffer_size() const;

        // deleted copy constructor and assignment operator
        RNG(RNG&) = delete;
        RNG& operator=(RNG&) = delete;
        RNG(RNG&&);
        RNG& operator=(RNG&&);

    private:
        std::unique_ptr<RNG_private> d_ptr_;
    };

    /**
     * @brief Returns memory used by all RNG instances.
     * 
     * @return size_t memory used by all RNG instances
     */
    const size_t memory_usage();
}
