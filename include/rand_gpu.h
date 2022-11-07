/**
 * @file rand_gpu.h
 * @author Jakob Kenda (kenda.jakob@gmail.com)
 * @brief 
 * @version 0.3
 * @date 2022-04-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef RAND_GPU_H
#define RAND_GPU_H


#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>


typedef void rand_gpu_rng;

/**
 * @brief random number generators
 */
enum rand_gpu_algorithm
{
    RAND_GPU_ALGORITHM_KISS09,
    RAND_GPU_ALGORITHM_LCG12864,
    RAND_GPU_ALGORITHM_LFIB,
    RAND_GPU_ALGORITHM_MRG63K3A,
    RAND_GPU_ALGORITHM_MSWS,
    RAND_GPU_ALGORITHM_MT19937,
    RAND_GPU_ALGORITHM_MWC64X,
    RAND_GPU_ALGORITHM_PCG6432,
    RAND_GPU_ALGORITHM_PHILOX2X32_10,
    RAND_GPU_ALGORITHM_RAN2,
    RAND_GPU_ALGORITHM_TINYMT64,
    RAND_GPU_ALGORITHM_TYCHE,
    RAND_GPU_ALGORITHM_TYCHE_I,
    RAND_GPU_ALGORITHM_WELL512,
    RAND_GPU_ALGORITHM_XORSHIFT6432STAR,
};

/**
 * @brief Initializes a new random number generator with default parameters.
 */
rand_gpu_rng *rand_gpu_new_default();

/**
 * @brief Initializes a new random number generator.
 * 
 * @param algorithm Algorithm for the RNG
 * @param n_buffers Number of buffers for storing random numbers
 * @param buffer_multi Buffer size multiplier
 */
rand_gpu_rng *rand_gpu_new(enum rand_gpu_algorithm algorithm, size_t n_buffers, size_t buffer_multi);

/**
 * @brief Initializes a new random number generator.
 * 
 * @param seed Custom seed
 * @param algorithm Algorithm for the RNG
 * @param n_buffers Number of buffers for storing random numbers
 * @param buffer_multi Buffer size multiplier
 */
rand_gpu_rng *rand_gpu_new_with_seed(uint64_t seed, enum rand_gpu_algorithm algorithm, size_t n_buffers, size_t buffer_multi);

/**
 * @brief Deletes the RNG.
 * @param rng RNG to be deleted
 */
void rand_gpu_delete(rand_gpu_rng *rng);


/**
 * @brief Delete all RNGs.
 */
void rand_gpu_delete_all();

/**
 * @brief Returns buffer size of RNG (equal for all RNGs).
 * @param rng RNG whose buffer size to retrieve
 */
size_t rand_gpu_rng_buffer_size(const rand_gpu_rng *rng);

/**
 * @brief Returns number of times the buffer was switched.
 * @param rng RNG whose buffer switches to retrieve
 */
size_t rand_gpu_rng_buffer_switches(const rand_gpu_rng *rng);

/**
 * @brief Returns number of times we had to wait for GPU to fill the buffer.
 * @param rng RNG whose buffer misses to retrieve
 */
size_t rand_gpu_rng_buffer_misses(const rand_gpu_rng *rng);

/**
 * @brief Returns RNG initialization time in ms.
 * @param rng RNG whose init time to retrieve
 */
float rand_gpu_rng_init_time(const rand_gpu_rng *rng);

/**
 * @brief Returns average calculation time for GPU is ms
 * @param rng RNG whose GPU calculation time to retrieve
 */
float rand_gpu_rng_avg_gpu_calculation_time(const rand_gpu_rng *rng);

/**
 * @brief Returns average transfer time for GPU is ms
 * @param rng RNG whose GPU transfer time to retrieve
 */
float rand_gpu_rng_avg_gpu_transfer_time(const rand_gpu_rng *rng);


/**
 * @brief Returns number of bytes occupied by all RNGs.
 */
size_t rand_gpu_memory_usage();

/**
 * @brief Returns number of times the buffer was switched in all RNG instances.
 */
size_t rand_gpu_buffer_switches();

/**
 * @brief Returns number of times we had to wait for GPU to fill the buffer in all RNG instances.
 */
size_t rand_gpu_buffer_misses();

/**
 * @brief Return average init time of all RNGs.
 */
float rand_gpu_avg_init_time();

/**
 * @brief Return average GPU transfer time of all RNGs.
 */
float rand_gpu_avg_gpu_calculation_time();

/**
 * @brief Return average GPU transfer time of all RNGs.
 */
float rand_gpu_avg_gpu_transfer_time();

/**
 * @brief Returns the compilation time for the algorithm in ms.
 * 
 * @param algorithm enum of the algorithm
 */
float rand_gpu_compilation_time(enum rand_gpu_algorithm algorithm);

/**
 * @brief Returns the name of the algorithm corresponding to the enum.
 * 
 * @param algorithm enum of the algorithm
 * @param long_name false - return only name, true - return full name and description
 */
const char *rand_gpu_algorithm_name(enum rand_gpu_algorithm algorithm, bool description);


/**
 * @brief Returns next random number.
 * @param rng RNG to retrieve the random number from
 */
uint64_t rand_gpu_u64(rand_gpu_rng *rng);

/**
 * @brief Returns next random number.
 * @param rng RNG to retrieve the random number from
 */
uint32_t rand_gpu_u32(rand_gpu_rng *rng);

/**
 * @brief Returns next random number.
 * @param rng RNG to retrieve the random number from
 */
uint16_t rand_gpu_u16(rand_gpu_rng *rng);

/**
 * @brief Returns next random number.
 * @param rng RNG to retrieve the random number from
 */
uint8_t rand_gpu_u8(rand_gpu_rng *rng);


/**
 * @brief Returns next random number.
 * @param rng RNG to retrieve the random number from
 */
float rand_gpu_float(rand_gpu_rng *rng);

/**
 * @brief Returns next random number.
 * @param rng RNG to retrieve the random number from
 */
double rand_gpu_double(rand_gpu_rng *rng);

/**
 * @brief Returns next random number.
 * @param rng RNG to retrieve the random number from
 */
long double rand_gpu_long_double(rand_gpu_rng *rng);


#ifdef __cplusplus
}
#endif

#endif
