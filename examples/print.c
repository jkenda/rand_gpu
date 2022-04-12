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

    puts("float:");
    for (int i = 0; i < 1000; i++) {
        float n = rand_gpu_float();
        printf("%f ", n);
    }
    printf("\n");

    puts("u64:");
    for (int i = 0; i < 1000; i++) {
        uint64_t n = rand_gpu_u64();
        printf("%lu ", n);
    }
    printf("\n");

    puts("i32:");
    for (int i = 0; i < 1000; i++) {
        uint32_t n = rand_gpu_i32();
        printf("%d ", n);
    }
    printf("\n");

    puts("u16:");
    for (int i = 0; i < 1000; i++) {
        uint16_t n = rand_gpu_u16();
        printf("%u ", n);
    }
    printf("\n");

    puts("i16:");
    for (int i = 0; i < 1000; i++) {
        int64_t n = rand_gpu_i16();
        printf("%ld ", n);
    }
    printf("\n");

    puts("u8:");
    for (int i = 0; i < 1000; i++) {
        uint8_t n = rand_gpu_u8();
        printf("%d ", n);
    }
    printf("\n");

    printf("memory usage: %lu\n", rand_gpu_bufsiz() * sizeof(uint32_t));
}
