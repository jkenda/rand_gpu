#include <iostream>
#include <cmath>
#include <chrono>
#include <random>
#include "../src/RNG.hpp"

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
    
    cout << "num. buffers: " << n_buffers << ", multi: " << multi << '\n';
    cout << "real pi: " << M_PI << '\n';

    auto start = system_clock::now();

    rand_gpu::RNG rng(n_buffers, multi);

    long cnt = 0;

    for (uint_fast64_t i = 0; i < SAMPLES; i++) {
        float a = rng.get_random<float>();
        float b = rng.get_random<float>();
        if (a*a + b*b < 1.0f) {
            cnt++;
        }
    }
    double pi_lib = (double) cnt / SAMPLES * 4;

    float time_lib = duration_cast<microseconds>(system_clock::now() - start).count() / (float) 1'000'000;
    printf("%lu misses\n", rng.buffer_misses());
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
