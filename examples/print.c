#include "../src/rand_gpu.h"
#include <stdio.h>

int main()
{
    rand_init();
    puts("float:");
    for (int i = 0; i < 4000; i++) {
        float n = rand_gpu_float();
        printf("%f ", n);
    }
    printf("\n");

    puts("double:");
    for (int i = 0; i < 4000; i++) {
        double n = rand_gpu_double();
        printf("%lf ", n);
    }
    printf("\n");

    puts("u16:");
    for (int i = 0; i < 4000; i++) {
        uint16_t n = rand_gpu_u16();
        printf("%u ", n);
    }
    printf("\n");

    puts("u32:");
    for (int i = 0; i < 4000; i++) {
        uint32_t n = rand_gpu_u32();
        printf("%u ", n);
    }
    printf("\n");

    puts("i64:");
    for (int i = 0; i < 4000; i++) {
        int64_t n = rand_gpu_i64();
        printf("%ld ", n);
    }
    printf("\n");

    rand_clean();
}
