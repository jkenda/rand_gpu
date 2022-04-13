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

#include <cstdio>
#include <vector>
#include "../src/RandGPU.hpp"

using std::vector;

int main()
{
    RandGPU r(4);

    size_t bufsiz = 2 * r.buffer_size();
    vector<int64_t> a;
    vector<int64_t> b;
    vector<int64_t> c;

    for (size_t i = 0; i < bufsiz; i++) {
        a[i] = r.rand<int64_t>();
    }

    for (size_t i = 0; i < bufsiz; i++) {
        b[i] = r.rand<int64_t>();
    }

    for (size_t i = 0; i < bufsiz; i++) {
        c[i] = r.rand<int64_t>();
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
