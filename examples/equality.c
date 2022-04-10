/**
 * @file test.c
 * @author Jakob Kenda (kenda.jakob@gmail.com)
 * @brief Tests whether the random number buffers stay the same over time
 * @version 0.1
 * @date 2022-04-09
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <stdio.h>
#include "../src/rand_gpu.h"

int main()
{
    rand_gpu64_init(16);

    size_t bufsiz = 2 * rand_gpu_bufsiz();
    int64_t a[bufsiz];
    int64_t b[bufsiz];
    int64_t c[bufsiz];

    for (size_t i = 0; i < bufsiz; i++) {
        a[i] = rand_gpu64_i64();
    }

    for (size_t i = 0; i < bufsiz; i++) {
        b[i] = rand_gpu64_i64();
    }

    for (size_t i = 0; i < bufsiz; i++) {
        c[i] = rand_gpu64_i64();
    }

    // test how similar they are
    long simab = 0;
    long simac = 0;
    long simbc = 0;
    for (size_t i = 0; i < bufsiz; i++) {
        if (a[i] == b[i]) simab++;
        if (b[i] == c[i]) simbc++;
        if (a[i] == c[i]) simac++;
    }

    printf("bufsiz: %lu\n", bufsiz);
    printf("similarity a <=> b: %lu / %lu, %f\n", simab, bufsiz, (float) simab / bufsiz);
    printf("similarity b <=> c: %lu / %lu, %f\n", simbc, bufsiz, (float) simbc / bufsiz);
    printf("similarity a <=> c: %lu / %lu, %f\n", simac, bufsiz, (float) simac / bufsiz);

    rand_gpu64_clean();
}
