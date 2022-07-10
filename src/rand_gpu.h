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
 * @brief Initializes a new random number generator.
 * 
 * @return New RNG with default parameters
 */
rand_gpu_rng *rand_gpu_new_default();

/**
 * @brief Initializes a new random number generator.
 * 
 * @param algorithm Algorithm for the RNG
 * @param n_buffers Number of buffers for storing random numbers
 * @param buffer_multi Buffer size multiplier
 * @return New RNG 
 */
rand_gpu_rng *rand_gpu_new(enum rand_gpu_algorithm algorithm, size_t n_buffers, size_t buffer_multi);

/**
 * @brief Initializes a new random number generator.
 * 
 * @param seed Custom seed
 * @param algorithm Algorithm for the RNG
 * @param n_buffers Number of buffers for storing random numbers
 * @param buffer_multi Buffer size multiplier
 * @return New RNG
 */
rand_gpu_rng *rand_gpu_new_with_seed(uint64_t seed, enum rand_gpu_algorithm algorithm, size_t n_buffers, size_t buffer_multi);

/**
 * @brief Deletes the RNG.
 * @param rng RNG to be deleted
 */
void rand_gpu_delete(rand_gpu_rng *rng);

/**
 * @brief Returns buffer size of RNG.
 * @param rng RNG whose buffer size to retrieve
 */
size_t rand_gpu_buffer_size(rand_gpu_rng *rng);

/**
 * @brief Returns number of times we had to wait for GPU to fill the buffer.
 * @param rng RNG whose misses to retrieve
 */
size_t rand_gpu_buf_misses(rand_gpu_rng *rng);

/**
 * @brief Returns RNG initialization time in ms.
 * @param rng RNG whose init time to retrieve
 */
float rand_gpu_init_time(rand_gpu_rng *rng);

/**
 * @brief Returns the name of the algorithm corresponding to the enum.
 * 
 * @param algorithm enum of the algorithm
 * @return const char* name of the algorithm
 */
const char *rand_gpu_algorithm_name(enum rand_gpu_algorithm algorithm, bool long_name);

/**
 * @brief Returns number of bytes occupied by all RNGs.
 * @return Allocated memory 
 */
size_t rand_gpu_memory();


/**
 * @brief Returns next random number.
 * @param rng RNG to retrieve the random number from
 */
unsigned long rand_gpu_u64(rand_gpu_rng *rng);

/**
 * @brief Returns next random number.
 * @param rng RNG to retrieve the random number from
 */
unsigned int rand_gpu_u32(rand_gpu_rng *rng);

/**
 * @brief Returns next random number.
 * @param rng RNG to retrieve the random number from
 */
unsigned short rand_gpu_u16(rand_gpu_rng *rng);

/**
 * @brief Returns next random number.
 * @param rng RNG to retrieve the random number from
 */
unsigned char rand_gpu_u8(rand_gpu_rng *rng);


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
