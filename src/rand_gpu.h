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


typedef void rand_gpu_rng;

/**
 * @brief Initializes a new random number generator.
 * @return New RNG
 */
rand_gpu_rng *rand_gpu_new(unsigned int multi);

/**
 * @brief Deletes the RNG.
 * @param rng RNG to be deleted
 */
void rand_gpu_delete(rand_gpu_rng *rng);

/**
 * @brief Returns buffer size of RNG
 * @param rng RNG whose buffer size to retrieve
 */
size_t rand_gpu_buffer_size(rand_gpu_rng *rng);

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
