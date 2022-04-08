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

/**
 * @brief Initialize connection to GPU,
 * 		  fill host and device buffers
 * 
 * @return Number of GPU threads (-1 if failed)
 */
int rand_init();

/**
 * @brief Cleans up, joins threads
 * 
 * @return Status (0 is OK)
 */
int rand_clean();

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
