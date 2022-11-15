#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "../include/rand_gpu.h"

#define SAMPLES (10000000000UL)

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

    printf("expected avg: 0.5\n");

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    rand_gpu_rng *rng = rand_gpu_new_rng(RAND_GPU_ALGORITHM_PHILOX2X32_10, n_buffers, multi);

    long cnt = 0;

    for (uint_fast64_t i = 0; i < SAMPLES; i++) {
        cnt += rand_gpu_bool(rng);
    }

    double avg = cnt / (double) SAMPLES;

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    float time_lib = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (float) (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;
    printf("%lu misses\n", rand_gpu_rng_buffer_misses(rng));
    printf("avg calculation time: %e ms\n", rand_gpu_rng_avg_gpu_calculation_time(rng) / (float) 1000);
    printf("avg transfer time:    %e ms\n", rand_gpu_rng_avg_gpu_transfer_time(rng) / (float) 1000);
    printf("lib avg ≃ %lf (+-%f), %f s\n", avg, ABS(avg - 0.5), time_lib);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    srand(time(NULL));
    cnt = 0;

    for (uint_fast64_t i = 0; i < SAMPLES; i++) {
        cnt += rand() % 2;
    }

    avg = cnt / (double) SAMPLES;

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    float time_std = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (float) (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;
    printf("std avg ≃ %lf (+-%f), %f s\n", avg, ABS(avg - 0.5), time_std);
    printf("speedup = %f\n", time_std / time_lib);

    rand_gpu_delete_rng(rng);
}
