/**
 * @file server.h
 * @author Jakob Kenda (kenda.jakob@gmail.com)
 * @brief 
 * @version 0.2
 * @date 2022-04-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include <stdint.h>
#include <stddef.h>


typedef int rand_gpu_rng;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initializes a new random number generator.
 * @return New RNG
 */
rand_gpu_rng rand_gpu_new(uint32_t multi);

/**
 * @brief Deletes the RNG.
 * @param rng RNG to be deleted
 */
void rand_gpu_delete(rand_gpu_rng rng);


/**
 * @brief Returns number of bytes occupied by all RNGs.
 * @return Allocated memory 
 */
size_t rand_gpu_memory();

/**
 * @brief Returns next random number.
 * @param rng from which RNG to retrieve the number
 */

uint64_t rand_gpu_u64(rand_gpu_rng rng);

/**
 * @brief Returns next random number.
 * @param rng from which RNG to retrieve the number
 */
int64_t  rand_gpu_i64(rand_gpu_rng rng);

/**
 * @brief Returns next random number.
 * @param rng from which RNG to retrieve the number
 */
uint32_t rand_gpu_u32(rand_gpu_rng rng);

/**
 * @brief Returns next random number.
 * @param rng from which RNG to retrieve the number
 */
int32_t  rand_gpu_i32(rand_gpu_rng rng);

/**
 * @brief Returns next random number.
 * @param rng from which RNG to retrieve the number
 */
uint16_t rand_gpu_u16(rand_gpu_rng rng);

/**
 * @brief Returns next random number.
 * @param rng from which RNG to retrieve the number
 */
int16_t  rand_gpu_i16(rand_gpu_rng rng);

/**
 * @brief Returns next random number.
 * @param rng from which RNG to retrieve the number
 */
uint8_t  rand_gpu_u8(rand_gpu_rng rng);

/**
 * @brief Returns next random number.
 * @param rng from which RNG to retrieve the number
 */
int8_t   rand_gpu_i8(rand_gpu_rng rng);


/**
 * @brief Returns next random number.
 * @param rng from which RNG to retrieve the number
 */
float       rand_gpu_float(rand_gpu_rng rng);

/**
 * @brief Returns next random number.
 * @param rng from which RNG to retrieve the number
 */
double      rand_gpu_double(rand_gpu_rng rng);

/**
 * @brief Returns next random number.
 * @param rng from which RNG to retrieve the number
 */
long double rand_gpu_long_double(rand_gpu_rng rng);


#ifdef __cplusplus
}
#endif
