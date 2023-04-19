#include <iostream>
#include <cmath>
#include <chrono>
#include <random>
#include "../include/RNG.hpp"

#define SAMPLES (5000000000UL)


using std::chrono::system_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

using namespace std;

int main(int argc, char **argv)
{
    size_t n_buffers = 4;
    size_t multi = 16;
    if (argc >= 2)
        sscanf(argv[1], "%lu", &n_buffers);
    if (argc == 3)
        sscanf(argv[2], "%lu", &multi);

    printf("real pi: %lf\n", M_PI);

    auto start = system_clock::now();

    rand_gpu_rng _rng = rand_gpu_new_rng(RAND_GPU_ALGORITHM_TYCHE, n_buffers, multi);
    rand_gpu::RNG rng0(_rng);
    rand_gpu::RNG<RAND_GPU_ALGORITHM_TINYMT64> rng1(n_buffers, multi);
    rng0.discard(rng0.buffer_size() / 2);

    long cnt = 0;

    for (uint_fast64_t i = 0; i < SAMPLES; i++) {
        float a = rng0.get_random<float>();
        float b = rng1.get_random<float>();
        if (a*a + b*b < 1.0f) {
            cnt++;
        }
    }
    double pi_lib = (double) cnt / SAMPLES * 4;

    float time_lib = duration_cast<microseconds>(system_clock::now() - start).count() / (float) 1'000'000;
    printf("%lu misses\n", rng0.buffer_misses() + rng1.buffer_misses());
    printf("avg calculation time: %e ms\n", (rng0.avg_gpu_calculation_time() + rng1.avg_gpu_calculation_time()).count() / (float) 1000);
    printf("avg transfer time:    %e ms\n", (rng0.avg_gpu_transfer_time() + rng1.avg_gpu_transfer_time()).count() / (float) 1000);
    printf("lib pi ≃ %lf (+-%f), %f s\n", pi_lib, abs(pi_lib - M_PI), time_lib);

    start = system_clock::now();

    mt19937 generator(duration_cast<microseconds>(system_clock::now().time_since_epoch()).count());
    cnt = 0;

    for (uint_fast64_t i = 0; i < SAMPLES; i++) {
        float a = generator() / (float) UINT32_MAX;
        float b = generator() / (float) UINT32_MAX;
        if (a*a + b*b < 1.0f) {
            cnt++;
        }
    }
    double pi_std = (double) cnt / SAMPLES * 4;

    float time_std = duration_cast<microseconds>(system_clock::now() - start).count() / (float) 1'000'000;
    printf("std pi ≃ %lf (+-%f), %f s\n", pi_std, abs(pi_std - M_PI), time_std);
    printf("speedup = %f\n", time_std / time_lib);
}
