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

double pi_std, pi_lib;
float time_std, time_lib;
struct timespec start, end;

int main()
{
    long cnt = 0;

    rand_gpu32_init();

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    for (uint32_t i = 0; i < SAMPLES; i++) {
        float a = rand_gpu32_float();
        float b = rand_gpu32_float();
        if (a*a + b*b < 1.0f) {
            cnt++;
        }
    }
    pi_lib = (double) cnt / SAMPLES * 4;

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    time_lib = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;
    printf("lib pi ≃ %lf (+-%f), %f s\n", pi_lib, ABS(pi_lib - M_PI), time_lib);

    rand_gpu32_clean();

    srand(time(NULL));
    cnt = 0;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    for (uint32_t i = 0; i < SAMPLES; i++) {
        float a = (float) rand() / RAND_MAX;
        float b = (float) rand() / RAND_MAX;
        if (a*a + b*b < 1.0f) {
            cnt++;
        }
    }
    pi_std = (double) cnt / SAMPLES * 4;

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    time_std = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;
    printf("std pi ≃ %lf (+-%f), %f s\n", pi_std, ABS(pi_std - M_PI), time_std);
    printf("speedup = %f\n", time_std / time_lib);
}
