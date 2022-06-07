#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "../src/rand_gpu.h"

#define SAMPLES (10000000000UL)

float abs_f(float a)
{
    if (a < 0) return -a;
    return a;
}

float pi_std()
{
    int nthreads = omp_get_max_threads();

    srand(time(NULL));
    uint64_t count = 0;

    omp_lock_t lock;
    omp_init_lock(&lock);

    #pragma omp parallel num_threads(nthreads)
    {
        uint_fast64_t cnt_l = 0;
        omp_set_lock(&lock);
        uint32_t seed = rand();
        omp_unset_lock(&lock);

        for (uint_fast64_t i = 0; i < SAMPLES / nthreads; i++) {
            float a = rand_r(&seed) / (float) RAND_MAX;
            float b = rand_r(&seed) / (float) RAND_MAX;
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
        rand_gpu_rng *rng = rand_gpu_new(4, 16);
        #pragma omp barrier

        uint_fast64_t cnt_l = 0;

        for (uint_fast64_t i = 0; i < SAMPLES / nthreads; i++) {
            float a = rand_gpu_float(rng);
            float b = rand_gpu_float(rng);
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
    size_t multi = 2;
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
