#define RAND_GPU_32
#include "../src/rand_gpu.h"
#include <stdio.h>
#include <limits.h>

int main()
{
    rand_gpu32_init();
    puts("float:");
    for (int i = 0; i < 4000; i++) {
        float n = rand_gpu32_float();
        printf("%f ", n);
    }
    printf("\n");

    puts("u16:");
    for (int i = 0; i < 4000; i++) {
        uint16_t n = rand_gpu32_u16();
        printf("%u ", n);
    }
    printf("\n");

    puts("u32:");
    for (int i = 0; i < 4000; i++) {
        uint32_t n = rand_gpu32_u32();
        printf("%u ", n);
    }
    printf("\n");

    puts("i16:");
    for (int i = 0; i < 4000; i++) {
        int64_t n = rand_gpu32_i16();
        printf("%ld ", n);
    }
    printf("\n");

    rand_gpu32_clean();

    printf("%.25e\n", 1.0/UINT_MAX);
    printf("%.25e\n", 1.0/ULONG_MAX);
}
