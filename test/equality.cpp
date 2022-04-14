/**
 * @file test.c
 * @author Jakob Kenda (kenda.jakob@gmail.com)
 * @brief Tests whether the random number buffers change over time
 *        Result should be 0.0 for a <=> b, a <=> c and b <=> c
 * @version 0.1
 * @date 2022-04-09
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <cstdio>
#include <vector>
#include "../src/RandGPU.hpp"

using std::vector;

int main()
{
    rand_gpu::RNG rng(4);

    size_t bufsiz = rng.buffer_size();
    vector<unsigned long> a;
    vector<unsigned long> b;
    vector<unsigned long> c;

    for (size_t i = 0; i < bufsiz; i++) {
        a[i] = rng.rand<unsigned long>();
    }

    for (size_t i = 0; i < bufsiz; i++) {
        b[i] = rng.rand<unsigned long>();
    }

    for (size_t i = 0; i < bufsiz; i++) {
        c[i] = rng.rand<unsigned long>();
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
}
