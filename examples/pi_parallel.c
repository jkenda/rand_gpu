#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "../include/rand_gpu.h"

#define SAMPLES (10000000000UL)

#define ABS(A) ((A) >= 0 ? (A) : -(A))


float pi_std(int nthreads)
{
    srand(time(NULL));
    uint64_t count = 0;

    omp_lock_t lock;
    omp_init_lock(&lock);

    #pragma omp parallel num_threads(nthreads)
    {
        uint_fast64_t cnt_l = 0;
        omp_set_lock(&lock);
        uint32_t state = rand();
        omp_unset_lock(&lock);

        for (uint_fast64_t i = 0; i < SAMPLES / nthreads; i++) {
            float a = rand_r(&state) / (float) RAND_MAX;
            float b = rand_r(&state) / (float) RAND_MAX;
            if (sqrt(a*a + b*b) < 1.0f)
                cnt_l++;
        }

        #pragma omp atomic
        count += cnt_l;
    }

    return (float) count / SAMPLES * 4;
}

float pi_lib(size_t n_buffers, size_t multi, int nthreads)
{
    uint64_t count = 0;

    #pragma omp parallel num_threads(nthreads)
    {
        rand_gpu_rng *rng = rand_gpu_new(RAND_GPU_ALGORITHM_TYCHE, n_buffers, multi);

        uint_fast64_t cnt_l = 0;

        for (uint_fast64_t i = 0; i < SAMPLES / nthreads; i++) {
            float a = rand_gpu_float(rng);
            float b = rand_gpu_float(rng);
            if (sqrt(a*a + b*b) < 1.0f)
                cnt_l++;
        }

        #pragma omp atomic
        count += cnt_l;

        printf("%d: %lu misses\n", omp_get_thread_num(), rand_gpu_buf_misses(rng));
    }

    return (float) count / SAMPLES * 4;
}

int main(int argc, char **argv)
{
    size_t n_buffers = 4;
    size_t multi = 16;
    int nthreads = omp_get_max_threads();
    if (argc >= 2)
        sscanf(argv[1], "%lu", &n_buffers);
    if (argc >= 3)
        sscanf(argv[2], "%lu", &multi);
    if (argc == 4)
        sscanf(argv[3], "%d", &nthreads);
    
    struct timespec start, end;
    float time_std, time_lib;

    printf("num. buffers: %lu, multi: %lu, nthreads: %d\n", n_buffers, multi, nthreads);
    printf("real pi: %lf\n", M_PI);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    float pi_l = pi_lib(n_buffers, multi, nthreads);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    time_lib = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;
    printf("lib pi: %f +/- %f, %f s\n", pi_l, ABS(M_PI - pi_l), time_lib);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    float pi_s = pi_std(nthreads);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    time_std = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;
    printf("std pi: %f +/- %f, %f s\n", pi_s, ABS(M_PI - pi_s), time_std);

    printf("speedup = %f\n", time_std / time_lib);
}
