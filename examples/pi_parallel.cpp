#include <cstdio>
#include <iostream>
#include <cmath>
#include <random>
#include <omp.h>
#include "../src/RNG.hpp"

#define SAMPLES (10000000000UL)

using namespace std;

float abs_f(float a)
{
    if (a < 0) return -a;
    return a;
}

float pi_std()
{
    int nthreads = omp_get_max_threads();

    uint64_t count = 0;
    random_device rd;
    mt19937 seed(rd());
    omp_lock_t lock;
    omp_init_lock(&lock);

    #pragma omp parallel num_threads(nthreads)
    {
        uint_fast64_t cnt_l = 0;
        omp_set_lock(&lock);
        mt19937 generator(seed());
        omp_unset_lock(&lock);

        for (uint_fast64_t i = 0; i < SAMPLES / nthreads; i++) {
            float a = generator() / (float) UINT32_MAX;
            float b = generator() / (float) UINT32_MAX;
            if (sqrt(a*a + b*b) < 1.0f)
                cnt_l++;
        }

        #pragma omp atomic
        count += cnt_l;
    }

    return (float) count / SAMPLES * 4;
}

float pi_lib(size_t multi)
{
    int nthreads = omp_get_max_threads();
    uint64_t count = 0;

    #pragma omp parallel num_threads(nthreads)
    {
        rand_gpu::RNG rng(multi);
        #pragma omp barrier
        #pragma omp single
        cout << rand_gpu::memory_usage() << '\n';

        uint_fast64_t cnt_l = 0;

        for (uint_fast64_t i = 0; i < SAMPLES / nthreads; i++) {
            float a = rng.get_random<float>();
            float b = rng.get_random<float>();
            if (sqrt(a*a + b*b) < 1.0f)
                cnt_l++;
        }

        #pragma omp atomic
        count += cnt_l;
    }

    return (float) count / SAMPLES * 4;
}

int main(int argc, char **argv)
{
    size_t multi = 16;
    if (argc == 2) {
        sscanf(argv[1], "%lu", &multi);
    }
    
    struct timespec start, end;
    float time_std, time_lib;

    printf("real pi: %lf\n", M_PI);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    float pi_l = pi_lib(multi);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    time_lib = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;
    printf("lib pi: %f +/- %f, %f s\n", pi_l, abs_f(M_PI - pi_l), time_lib);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    float pi_s = pi_std();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    time_std = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;
    printf("std pi: %f +/- %f, %f s\n", pi_s, abs_f(M_PI - pi_s), time_std);

    printf("speedup = %f\n", time_std / time_lib);
}
