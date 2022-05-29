#include "../src/RNG.hpp"
#include <iostream>
#include <cmath>
#include <chrono>

#define SAMPLES (1000000000UL)

double pi_std, pi_lib;
float time_std, time_lib;
struct timespec start, end;

using std::chrono::system_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

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
    
    auto start = system_clock::now();

    rand_gpu::RNG rng(times);

    long cnt = 0;

    for (uint_fast64_t i = 0; i < SAMPLES; i++) {
        float a = rng.get_random<float>();
        float b = rng.get_random<float>();
        if (a*a + b*b < 1.0f) {
            cnt++;
        }
    }
    pi_lib = (double) cnt / SAMPLES * 4;

    time_lib = duration_cast<microseconds>(system_clock::now() - start).count() / (float) 1'000'000;
    printf("lib pi ≃ %lf (+-%f), %f s\n", pi_lib, abs_f(pi_lib - M_PI), time_lib);

    start = system_clock::now();

    srand(time(NULL));
    cnt = 0;

    for (uint_fast64_t i = 0; i < SAMPLES; i++) {
        float a = (float) rand() / RAND_MAX;
        float b = (float) rand() / RAND_MAX;
        if (a*a + b*b < 1.0f) {
            cnt++;
        }
    }
    pi_std = (double) cnt / SAMPLES * 4;

    time_std = duration_cast<microseconds>(system_clock::now() - start).count() / (float) 1'000'000;
    printf("std pi ≃ %lf (+-%f), %f s\n", pi_std, abs_f(pi_std - M_PI), time_std);
    printf("speedup = %f\n", time_std / time_lib);
    printf("%lu misses\n", rng.buffer_misses());
}
