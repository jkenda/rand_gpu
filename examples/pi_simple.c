#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>

#include "../src/rand_gpu.h"

#define SAMPLES (100000000UL)

float pi_lib;
float time_lib;
struct timespec start, end;

float abs_f(float a)
{
    if (a < 0) return -a;
    return a;
}


int main(int argc, char **argv)
{
    size_t times = 16;
    if (argc == 2) {
        sscanf(argv[1], "%lu", &times);
    }
    
    rand_gpu_rng *rng = rand_gpu_new(4, times);

    long cnt = 0;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    for (uint32_t i = 0; i < SAMPLES; i++) {
        float a = rand_gpu_float(rng);
        float b = rand_gpu_float(rng);
        if (a*a + b*b < 1.0f) {
            cnt++;
        }
    }
    pi_lib = (double) cnt / SAMPLES * 4;

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    time_lib = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;
    printf("lib pi ≃ %f (+-%f), %f s\n", pi_lib, abs_f(pi_lib - M_PI), time_lib);
}
