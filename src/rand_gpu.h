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

/**
 * @brief Initialize the library -
 * 		  this function has to be called before any random numbers can be retrieved
 * 
 * @return Sum of all statuses
 */
int rand_gpu_init();

/**
 * @brief Uninitialize the library -
 *        this function is to be called after all other rand_gpu functions
 * 
 * @return Sum of all return statuses 
 */
int rand_gpu_clean();

size_t rand_gpu_bufsiz();

uint64_t rand_gpu_u64();
int64_t rand_gpu_i64();
uint32_t rand_gpu_u32();
int32_t rand_gpu_i32();
uint16_t rand_gpu_u16();
int16_t rand_gpu_i16();

long rand_gpu_long();
unsigned long rand_gpu_unsigned_long();
int rand_gpu_int();
unsigned int rand_gpu_unsigned_int();
short rand_gpu_short();
unsigned short rand_gpu_ushort();
float rand_gpu_float();
double rand_gpu_double();
