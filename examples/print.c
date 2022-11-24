#include <stdio.h>
#include <limits.h>
#include <stdint.h>
#include "../include/rand_gpu.h"

int main()
{
    rand_gpu_rng rng = rand_gpu_new_rng_default();

    printf("u8: %lu bits\n", 8 * sizeof(uint8_t));
    for (int i = 0; i < 500; i++) {
        uint8_t n = rand_gpu_rng_get_random_8b(rng);
        printf("%u ", n);
    }
    printf("\n\n");

    printf("i8: %lu bits\n", 8 * sizeof(int8_t));
    for (int i = 0; i < 500; i++) {
        int8_t n = rand_gpu_rng_get_random_8b(rng);
        printf("%d ", n);
    }
    printf("\n\n");

    printf("u16: %lu bits\n", 8 * sizeof(uint16_t));
    for (int i = 0; i < 500; i++) {
        uint16_t n = rand_gpu_rng_get_random_16b(rng);
        printf("%u ", n);
    }
    printf("\n\n");

    printf("i16: %lu bits\n", 8 * sizeof(int16_t));
    for (int i = 0; i < 500; i++) {
        int16_t n = rand_gpu_rng_get_random_16b(rng);
        printf("%d ", n);
    }
    printf("\n\n");

    printf("u32: %lu bits\n", 8 * sizeof(uint32_t));
    for (int i = 0; i < 500; i++) {
        uint32_t n = rand_gpu_rng_get_random_32b(rng);
        printf("%u ", n);
    }
    printf("\n\n");

    printf("i32: %lu bits\n", 8 * sizeof(int32_t));
    for (int i = 0; i < 500; i++) {
        int32_t n = rand_gpu_rng_get_random_32b(rng);
        printf("%d ", n);
    }
    printf("\n\n");

    printf("u64: %lu bits\n", 8 * sizeof(uint64_t));
    for (int i = 0; i < 500; i++) {
        uint64_t n = rand_gpu_rng_get_random_64b(rng);
        printf("%lu ", n);
    }
    printf("\n\n");

    printf("i64: %lu bits\n", 8 * sizeof(int64_t));
    for (int i = 0; i < 500; i++) {
        int64_t n = rand_gpu_rng_get_random_64b(rng);
        printf("%ld ", n);
    }
    printf("\n\n");

    printf("float: %lu bits\n", 8 * sizeof(float));
    for (int i = 0; i < 500; i++) {
        float n = rand_gpu_get_random_float(rng);
        printf("%f ", n);
    }
    printf("\n\n");

    printf("double: %lu bits\n", 8 * sizeof(double));
    for (int i = 0; i < 500; i++) {
        double n = rand_gpu_get_random_double(rng);
        printf("%lf ", n);
    }
    printf("\n\n");

    printf("long double: %lu bits\n", 8 * sizeof(long double));
    for (int i = 0; i < 500; i++) {
        long double n = rand_gpu_get_random_long_double(rng);
        printf("%Lf ", n);
    }
    printf("\n\n");

    printf("memory usage: %lu bytes\n", rand_gpu_memory_usage());
}
