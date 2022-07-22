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
static const size_t N_ALGORITHMS = RAND_GPU_ALGORITHM_XORSHIFT6432STAR - RAND_GPU_ALGORITHM_KISS09 + 1;

struct perf
{
    enum rand_gpu_algorithm algorithm;
    int n_buffers, multi;
    float speedup;
};

int cmp_perf(const void *a, const void *b) {
    struct perf *ap = (struct perf *) a;
    struct perf *bp = (struct perf *) b;
    return ap->speedup < bp->speedup ? 1 : -1;
}

float get_time(enum rand_gpu_algorithm algorithm, int n_buffers, int multi)
{
    struct timespec start, end;

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
    struct timespec start, end;

    srand(time(NULL));

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    for (uint32_t i = 0; i < SAMPLES; i++) {
        rand() / (float) RAND_MAX;
        rand() / (float) RAND_MAX;
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    float time_std = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000) / 1000000;

    struct perf best_perf[N_ALGORITHMS];

    for (int algorithm = RAND_GPU_ALGORITHM_KISS09; algorithm <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; algorithm++)
    {
        float best_speedup = 0;
        int fastest_n_buffers, fastest_multi;

        puts("algorithm,multi,2,4,8,16");

        for (int multi = 1; multi <= 64; multi *= 2)
        {
            printf("%s,%02d", rand_gpu_algorithm_name(algorithm, false), multi);
            fflush(stdout);

            for (int n_buffers = 2; n_buffers <= 16; n_buffers *= 2)
            {
                float time_lib = get_time(algorithm, n_buffers, multi);
                float speedup = time_std / time_lib;

                printf(",%.5f", speedup);
                fflush(stdout);

                if (speedup > best_speedup)
                {
                    best_speedup = speedup;
                    fastest_n_buffers = n_buffers;
                    fastest_multi = multi;
                }
            }
            printf("\n");
        }
        best_perf[algorithm].algorithm = algorithm;
        best_perf[algorithm].n_buffers = fastest_n_buffers;
        best_perf[algorithm].multi = fastest_multi;
        best_perf[algorithm].speedup = best_speedup;
        printf("\n");
    }

    qsort(best_perf, N_ALGORITHMS, sizeof(struct perf), cmp_perf);

    puts("\nalgorithm,n_buffers,multi,speedup");
    for (int i = 0; i < N_ALGORITHMS; i++)
    {
        printf("%s,%d,%d,%f\n", rand_gpu_algorithm_name(best_perf[i].algorithm, false), 
            best_perf[i].n_buffers, best_perf[i].multi, best_perf[i].speedup);
    }
}
