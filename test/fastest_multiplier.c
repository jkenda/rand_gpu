#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>

#define RAND_GPU_32
#include "../src/rand_gpu.h"

#define ABS(A) ((A >= 0) ? (A) : -(A))
#define SAMPLES (1000000000UL)

struct timespec start, end;

float get_time(uint32_t multi)
{
    rand_gpu_rng *rng = rand_gpu_new(4, multi, RAND_GPU_ALGORITHM_TYCHE_I);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    for (uint32_t i = 0; i < SAMPLES; i++) {
        rand_gpu_float(rng);
        rand_gpu_float(rng);
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    rand_gpu_delete(rng);

    float time = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;

    return time;
}

int main()
{
    srand(time(NULL));

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    for (uint32_t i = 0; i < SAMPLES; i++) {
        rand() / (float) RAND_MAX;
        rand() / (float) RAND_MAX;
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    float time_std = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;

    puts("multi,time,speedup");
    for (int i = 1; i <= 128; i *= 2) {
        float time_lib = get_time(i);
        printf("%d,%f,%f\n", i, time_lib, time_std / time_lib);
    }
}
