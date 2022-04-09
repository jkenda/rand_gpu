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

#ifdef RAND_GPU_32

/**
 * @brief Initialize the library -
 * 		  this function has to be called before any random numbers can be retrieved
 * 
 * @return Sum of all statuses
 */
int rand_gpu32_init();

/**
 * @brief Uninitialize the library -
 *        this function is to be called after all other rand_gpu functions
 * 
 */
void rand_gpu32_clean();

uint32_t rand_gpu32_u32();
int32_t rand_gpu32_i32();
uint16_t rand_gpu32_u16();
int16_t rand_gpu32_i16();

int rand_gpu32_int();
unsigned int rand_gpu32_uint();
short rand_gpu32_short();
unsigned short rand_gpu32_ushort();
float rand_gpu32_float();

#else

/**
 * @brief Initialize the library -
 * 		  this function has to be called before any random numbers can be retrieved
 * 
 * @return Sum of all statuses
 */
int rand_gpu64_init();

/**
 * @brief Uninitialize the library -
 *        this function is to be called after all other rand_gpu functions
 * 
 */
void rand_gpu64_clean();

uint64_t rand_gpu64_u64();
int64_t rand_gpu64_i64();
uint32_t rand_gpu64_u32();
int32_t rand_gpu64_i32();
uint16_t rand_gpu64_u16();
int16_t rand_gpu64_i16();

long rand_gpu64_long();
unsigned long rand_gpu64_ulong();
int rand_gpu64_int();
unsigned int rand_gpu64_unsigned_int();
short rand_gpu64_short();
unsigned short rand_gpu64_ushort();
float rand_gpu64_float();
double rand_gpu64_double();

#endif

size_t rand_gpu_bufsiz();
