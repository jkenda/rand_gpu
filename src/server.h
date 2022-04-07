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

uint64_t rand_get_u64();
int64_t rand_get_i64();
uint32_t rand_get_u32();
int32_t rand_get_i32();
uint16_t rand_get_u16();
int16_t rand_get_i16();

long rand_get_long();
unsigned long rand_get_unsigned_long();
int rand_get_int();
unsigned int rand_get_unsigned_int();
short rand_get_short();
unsigned short rand_get_ushort();
float rand_get_float();
double rand_get_double();
