#include <iostream>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <random>
#include "../src/RNG.hpp"

#define SAMPLES (100'000'000UL)

using namespace std;
using std::chrono::system_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

int main(int argc, char **argv)
{
    size_t n_buffers = 8;
    size_t multi = 16;
    if (argc >= 2)
        sscanf(argv[1], "%lu", &n_buffers);
    if (argc == 3)
        sscanf(argv[2], "%lu", &multi);
    
    auto start = system_clock::now();

    rand_gpu::RNG rng(n_buffers, multi);

    long cnt = 0;

    for (uint32_t i = 0; i < SAMPLES; i++) {
        float a = rng.get_random<float>();
        float b = rng.get_random<float>();
        if (a*a + b*b < 1.0f) {
            cnt++;
        }
    }
    double pi = (double) cnt / SAMPLES * 4;

    float time = duration_cast<microseconds>(system_clock::now() - start).count() / (float) 1'000'000;

    cout << "lib pi ≃ " << pi << " (+-" << abs(pi - M_PI) << "), " << time << " s\n";
    cout << rng.buffer_misses() << " misses\n";
}
