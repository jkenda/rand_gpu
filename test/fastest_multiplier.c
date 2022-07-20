#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include <limits.h>

#define RAND_GPU_32
#include "../include/rand_gpu.h"

#define ABS(A) ((A >= 0) ? (A) : -(A))
#define SAMPLES (1000000000UL)

struct timespec start, end;

float get_time(enum rand_gpu_algorithm algorithm, int n_buffers, int multi)
{
    rand_gpu_rng *rng = rand_gpu_new(algorithm, n_buffers, multi);

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

    float fastest_time = __FLT_MAX__;
    enum rand_gpu_algorithm fastest_algorithm = RAND_GPU_ALGORITHM_KISS09;
    int fastest_n_buffers, fastest_multi;

    puts("algorithm,n_buffers,multi,time,speedup");
    for (int algorithm = RAND_GPU_ALGORITHM_KISS09; algorithm <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; algorithm++)
    {
        for (int n_buffers = 2; n_buffers <= 16; n_buffers *= 2)
        {
            for (int multi = 1; multi <= 64; multi *= 2)
            {
                float time_lib = get_time(algorithm, n_buffers, multi);
                printf("%s,%d,%d,%f,%f\n", rand_gpu_algorithm_name(algorithm, false), n_buffers, multi, 
                    time_lib, time_std / time_lib);
                fflush(stdout);
                if (time_lib < fastest_time)
                {
                    fastest_time = time_lib;
                    fastest_algorithm = algorithm;
                    fastest_n_buffers = n_buffers;
                    fastest_multi = multi;
                }
            }
        }
    }

    puts("\nfastest:");
    printf("%s,%d,%d", rand_gpu_algorithm_name(fastest_algorithm, false), fastest_n_buffers, fastest_multi);
}
