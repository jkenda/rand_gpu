#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <omp.h>
#include "../include/rand_gpu.h"

#define SAMPLES (10000000000UL)

#define ABS(A) ((A) >= 0 ? (A) : -(A))


float pi_std()
{
    srand(time(NULL));
    uint64_t count = 0;

    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

    #pragma omp parallel
    {
        uint_fast64_t cnt_l = 0;
        pthread_mutex_lock(&lock);
        uint32_t state = rand();
        pthread_mutex_unlock(&lock);

        #pragma omp for schedule(static)
        for (uint_fast64_t i = 0; i < SAMPLES; i++) {
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

float pi_lib(size_t n_buffers, size_t multi)
{
    uint64_t count = 0;

    #pragma omp parallel
    {
        rand_gpu_rng rng = rand_gpu_new_rng(RAND_GPU_ALGORITHM_PCG6432, n_buffers, multi);
        rand_gpu_rng_discard(rng, rand_gpu_rng_buffer_size(rng) * omp_get_thread_num() / omp_get_num_threads());

        uint_fast64_t cnt_l = 0;

        #pragma omp for schedule(static)
        for (uint_fast64_t i = 0; i < SAMPLES; i++) {
            float a = rand_gpu_get_random_float(rng);
            float b = rand_gpu_get_random_float(rng);
            if (sqrt(a*a + b*b) < 1.0f)
                cnt_l++;
        }

        #pragma omp atomic
        count += cnt_l;

        printf("%lu misses\n", rand_gpu_rng_buffer_misses(rng));
    }

    rand_gpu_delete_all();

    return (float) count / SAMPLES * 4;
}

int main(int argc, char **argv)
{
    size_t n_buffers = 4;
    size_t multi = 16;

    if (argc >= 2)
        sscanf(argv[1], "%lu", &n_buffers);
    if (argc >= 3)
        sscanf(argv[2], "%lu", &multi);
    
    struct timespec start, end;
    float time_std, time_lib;

    printf("num. buffers: %lu, multi: %lu\n", n_buffers, multi);
    printf("real pi: %lf\n", M_PI);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    float pi_l = pi_lib(n_buffers, multi);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    time_lib = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;
    printf("lib pi: %f (+/- %f), %f s\n", pi_l, ABS(M_PI - pi_l), time_lib);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    float pi_s = pi_std();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    time_std = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;
    printf("std pi: %f (+/- %f), %f s\n", pi_s, ABS(M_PI - pi_s), time_std);

    printf("speedup = %f\n", time_std / time_lib);
}
