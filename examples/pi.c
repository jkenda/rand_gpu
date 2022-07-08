#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "../src/rand_gpu.h"

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

    printf("num. buffers: %lu, multi: %lu\n", n_buffers, multi);
    printf("real pi: %lf\n", M_PI);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    rand_gpu_rng *rng = rand_gpu_new(n_buffers, multi, RAND_GPU_ALGORITHM_MT19937);

    long cnt = 0;

    for (unsigned long i = 0; i < SAMPLES; i++) {
        float a = rand_gpu_float(rng);
        float b = rand_gpu_float(rng);
        if (a*a + b*b < 1.0f) {
            cnt++;
        }
    }
    double pi_lib = (double) cnt / SAMPLES * 4;

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    rand_gpu_delete(rng);

    float time_lib = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;
    printf("%lu misses\n", rand_gpu_buf_misses(rng));
    printf("lib pi ≃ %lf (+-%f), %f s\n", pi_lib, ABS(pi_lib - M_PI), time_lib);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    srand(time(NULL));
    cnt = 0;

    for (unsigned long i = 0; i < SAMPLES; i++) {
        float a = (float) rand() / RAND_MAX;
        float b = (float) rand() / RAND_MAX;
        if (a*a + b*b < 1.0f) {
            cnt++;
        }
    }
    double pi_std = (double) cnt / SAMPLES * 4;

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    float time_std = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;
    printf("std pi ≃ %lf (+-%f), %f s\n", pi_std, ABS(pi_std - M_PI), time_std);
    printf("speedup = %f\n", time_std / time_lib);
}
