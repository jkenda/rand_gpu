#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#include <inttypes.h>
#include "../src/rand_gpu.h"

#define NTHREADS 8
#define ITER (2048U * 1024U)

float abs_f(float a)
{
    if (a < 0) return -a;
    return a;
}

float pi_std()
{
    uint32_t states[NTHREADS];
    srand(time(NULL));
    states[0] = rand();
    for (uint32_t i = 1; i < NTHREADS; i++) {
        states[i] = rand();
    }

    uint64_t count = 0;

    #pragma omp parallel num_threads(NTHREADS)
    {
        uint32_t thread_num = omp_get_thread_num();
        uint_fast64_t cnt_l = 0;

        for (uint32_t i = 0; i < ITER; i++) {
            float a = rand_r(&states[thread_num]) / (float) RAND_MAX;
            float b = rand_r(&states[thread_num]) / (float) RAND_MAX;
            if (sqrt(a*a + b*b) < 1.0f)
                cnt_l++;
        }

        #pragma omp atomic
        count += cnt_l;
    }

    return (float) count / (ITER * NTHREADS) * 4;
}

float pi_lib()
{
    uint64_t count = 0;

    #pragma omp parallel num_threads(NTHREADS)
    {
        rand_gpu_rng *rng = rand_gpu_new(2);
        uint_fast64_t cnt_l = 0;

        for (uint32_t i = 0; i < ITER; i++) {
            float a = rand_gpu_float(rng);
            float b = rand_gpu_float(rng);
            if (sqrt(a*a + b*b) < 1.0f)
                cnt_l++;
        }

        #pragma omp atomic
        count += cnt_l;
        rand_gpu_delete(rng);
    }

    return (float) count / (ITER * NTHREADS) * 4;
}

int main()
{
    struct timespec start, end;
    float time_std, time_lib;

    printf("real pi: %lf\n", M_PI);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    float pi_s = pi_std();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    time_std = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;

    printf("std pi: %f +/- %f, %f s\n", pi_s, abs_f(M_PI - pi_s), time_std);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    float pi_l = pi_lib();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    time_lib = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;
    printf("lib pi: %f +/- %f, %f s\n", pi_l, abs_f(M_PI - pi_l), time_lib);
    printf("speedup = %f\n", time_std / time_lib);
}