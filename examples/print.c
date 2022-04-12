#define RAND_GPU_32
#include "../src/rand_gpu.h"
#include <stdio.h>
#include <limits.h>

int main(int argc, char **argv)
{
    size_t times = 16;
    if (argc == 2) {
        sscanf(argv[1], "%lu", &times);
    }
    
    rand_gpu_init(times);

    printf("u8: %lu bits\n", 8 * sizeof(uint8_t));
    for (int i = 0; i < 100; i++) {
        uint8_t n = rand_gpu_u8();
        printf("%u ", n);
    }
    printf("\n");

    printf("i8: %lu bits\n", 8 * sizeof(int8_t));
    for (int i = 0; i < 100; i++) {
        int8_t n = rand_gpu_i8();
        printf("%d ", n);
    }
    printf("\n");

    printf("u16: %lu bits\n", 8 * sizeof(uint16_t));
    for (int i = 0; i < 100; i++) {
        uint16_t n = rand_gpu_u16();
        printf("%u ", n);
    }
    printf("\n");

    printf("i16: %lu bits\n", 8 * sizeof(int16_t));
    for (int i = 0; i < 100; i++) {
        int16_t n = rand_gpu_i16();
        printf("%d ", n);
    }
    printf("\n");

    printf("u32: %lu bits\n", 8 * sizeof(uint32_t));
    for (int i = 0; i < 100; i++) {
        uint32_t n = rand_gpu_u32();
        printf("%u ", n);
    }
    printf("\n");

    printf("i32: %lu bits\n", 8 * sizeof(int32_t));
    for (int i = 0; i < 100; i++) {
        int32_t n = rand_gpu_i32();
        printf("%d ", n);
    }
    printf("\n");

    printf("u64: %lu bits\n", 8 * sizeof(uint64_t));
    for (int i = 0; i < 100; i++) {
        uint64_t n = rand_gpu_u64();
        printf("%lu ", n);
    }
    printf("\n");

    printf("i64: %lu bits\n", 8 * sizeof(int64_t));
    for (int i = 0; i < 100; i++) {
        int64_t n = rand_gpu_i64();
        printf("%ld ", n);
    }
    printf("\n");

    printf("float: %lu bits\n", 8 * sizeof(float));
    for (int i = 0; i < 100; i++) {
        float n = rand_gpu_float();
        printf("%f ", n);
    }
    printf("\n");

    printf("double: %lu bits\n", 8 * sizeof(double));
    for (int i = 0; i < 100; i++) {
        double n = rand_gpu_double();
        printf("%lf ", n);
    }
    printf("\n");

    printf("long double: %lu bits\n", 8 * sizeof(long double));
    for (int i = 0; i < 100; i++) {
        long double n = rand_gpu_long_double();
        printf("%Lf ", n);
    }
    printf("\n");

    printf("memory usage: %lu bytes\n", rand_gpu_bufsiz() * sizeof(uint32_t));
}
