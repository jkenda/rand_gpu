#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "../include/rand_gpu.h"

#define SAMPLES (5000000000UL)

#define ABS(A) ((A) >= 0 ? (A) : -(A))


struct timespec start, end;

int main(int argc, char **argv)
{
    size_t n_buffers = 4;
    size_t multi = 16;
    if (argc >= 2)
        sscanf(argv[1], "%lu", &n_buffers);
    if (argc == 3)
        sscanf(argv[2], "%lu", &multi);

    printf("real pi: %lf\n", M_PI);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    rand_gpu_rng rng0 = rand_gpu_new_rng(RAND_GPU_ALGORITHM_MT19937, n_buffers, multi);
    rand_gpu_rng rng1 = rand_gpu_new_rng(RAND_GPU_ALGORITHM_MT19937, n_buffers, multi);
    rand_gpu_rng_discard(rng0, rand_gpu_rng_buffer_size(rng0) / (n_buffers * 2));
    
    long cnt = 0;

    for (uint_fast64_t i = 0; i < SAMPLES; i++) {
        float a = rand_gpu_get_random_float(rng0);
        float b = rand_gpu_get_random_float(rng1);
        if (a*a + b*b < 1.0f) {
            cnt++;
        }
    }
    double pi_lib = (double) cnt / SAMPLES * 4;

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    float time_lib = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (float) (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;
    printf("%lu misses\n", rand_gpu_rng_buffer_misses(rng0) + rand_gpu_rng_buffer_misses(rng1));
    printf("avg calculation time: %e ms\n", (rand_gpu_rng_avg_gpu_calculation_time(rng0) + rand_gpu_rng_avg_gpu_calculation_time(rng1)) / (float) 1000);
    printf("avg transfer time:    %e ms\n", (rand_gpu_rng_avg_gpu_transfer_time(rng0) + rand_gpu_rng_avg_gpu_transfer_time(rng1)) / (float) 1000);
    printf("lib pi ≃ %lf (+-%f), %f s\n", pi_lib, ABS(pi_lib - M_PI), time_lib);
    fflush(stdout);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    srand(time(NULL));
    cnt = 0;

    for (uint_fast64_t i = 0; i < SAMPLES; i++) {
        float a = rand() / (float) RAND_MAX;
        float b = rand() / (float) RAND_MAX;
        if (a*a + b*b < 1.0f) {
            cnt++;
        }
    }
    double pi_std = (double) cnt / SAMPLES * 4;

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    float time_std = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;
    printf("std pi ≃ %lf (+-%f), %f s\n", pi_std, ABS(pi_std - M_PI), time_std);
    printf("speedup = %f\n", time_std / time_lib);

    rand_gpu_delete_all();
}
