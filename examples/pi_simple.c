#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include "../src/rand_gpu.h"

#define SAMPLES (100000000UL)
#define ABS(A) ((A) >= 0 ? (A) : -(A))

struct timespec start, end;

int main(int argc, char **argv)
{
    size_t n_buffers = 8;
    size_t multi = 16;
    if (argc >= 2)
        sscanf(argv[1], "%lu", &n_buffers);
    if (argc == 3)
        sscanf(argv[2], "%lu", &multi);
    
    printf("num. buffers: %lu, multi: %lu\n", n_buffers, multi);
    printf("real pi: %lf\n", M_PI);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    rand_gpu_rng *rng = rand_gpu_new(RAND_GPU_ALGORITHM_TYCHE, n_buffers, multi);

    long cnt = 0;

    for (uint32_t i = 0; i < SAMPLES; i++) {
        float a = rand_gpu_float(rng);
        float b = rand_gpu_float(rng);
        if (a*a + b*b < 1.0f) {
            cnt++;
        }
    }
    float pi = (double) cnt / SAMPLES * 4;

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    float time = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;
    printf("%lu misses\n", rand_gpu_buf_misses(rng));
    printf("lib pi ≃ %f (+-%f), %f s\n", pi, ABS(pi - M_PI), time);
}
