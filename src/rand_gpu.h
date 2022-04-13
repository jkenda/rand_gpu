/**
 * @file server.h
 * @author Jakob Kenda (kenda.jakob@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-04-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#pragma once

#include <stdint.h>
#include <stddef.h>


typedef int rand_gpu_rng;

/**
 * @brief Initialize the 64-bit version of the library -
 * 		  this function has to be called before any random numbers can be retrieved
 * 
 * @return Sum of all statuses
 */
rand_gpu_rng rand_gpu_new(uint32_t multi);

void rand_gpu_delete(rand_gpu_rng rng);


size_t rand_gpu_memory();


uint64_t rand_gpu_u64(rand_gpu_rng rng);
int64_t  rand_gpu_i64(rand_gpu_rng rng);
uint32_t rand_gpu_u32(rand_gpu_rng rng);
int32_t  rand_gpu_i32(rand_gpu_rng rng);
uint16_t rand_gpu_u16(rand_gpu_rng rng);
int16_t  rand_gpu_i16(rand_gpu_rng rng);
uint8_t  rand_gpu_u8(rand_gpu_rng rng);
int8_t   rand_gpu_i8(rand_gpu_rng rng);

float       rand_gpu_float(rand_gpu_rng rng);
double      rand_gpu_double(rand_gpu_rng rng);
long double rand_gpu_long_double(rand_gpu_rng rng);
