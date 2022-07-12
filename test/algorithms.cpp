#include "../include/RNG.hpp"
#include <iostream>
#include <cstdint>
#include <cmath>
#include <chrono>

#define SAMPLES (1000000000UL)

using std::chrono::system_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

using namespace std;

template <int I>
void print()
{
    print<I-1>();

    constexpr rand_gpu_algorithm algorithm = static_cast<rand_gpu_algorithm>(I);
    rand_gpu::RNG<algorithm> rng(0, 2, 1);
    cout << rand_gpu::algorithm_name(algorithm, true) << ":\n";
    cout << "init time: " << rng.init_time() << " ms\n";

    for (size_t j = 0; j < 256; j++)
    {
        cout << std::to_string(rng.template get_random<uint8_t>()) << ' ';
    }
    cout << "\n\n";
}

template<>
void print<-1>()
{
}

template <int I>
void time_pi(float time_std)
{
    time_pi<I-1>(time_std);

    constexpr rand_gpu_algorithm algorithm = static_cast<rand_gpu_algorithm>(I);
    rand_gpu::RNG<algorithm> rng(0, 8, 32);

    auto start = system_clock::now();
    long cnt = 0;

    for (uint_fast64_t i = 0; i < SAMPLES; i++) {
        float a = rng.template get_random<float>();
        float b = rng.template get_random<float>();
        if (a*a + b*b < 1.0f) {
            cnt++;
        }
    }
    float pi = cnt / (double) SAMPLES * 4;

    float time_lib = duration_cast<microseconds>(system_clock::now() - start).count() / (float) 1'000'000;
    cout << "pi ≃ " << pi << " (+-" << abs(pi - M_PI) << "), ";
    cout << "speedup = " << time_std / time_lib << ", ";
    cout << rng.buffer_misses() << " misses - ";
    cout << rand_gpu::algorithm_name(algorithm) << '\n';
}

template <>
void time_pi<-1>(float time_std)
{
}

int main()
{
    print<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>();

    long cnt = 0;

    srand(0);

    auto start = system_clock::now();

    for (uint_fast64_t i = 0; i < SAMPLES; i++) {
        float a = rand() / (float) RAND_MAX;
        float b = rand() / (float) RAND_MAX;
        if (a*a + b*b < 1.0f) {
            cnt++;
        }
    }
    double pi = (double) cnt / SAMPLES * 4;
    cout << "pi ≃ " << pi << " (+-" << abs(pi - M_PI) << ")\n";

    float time_std = duration_cast<microseconds>(system_clock::now() - start).count() / (float) 1'000'000;

    time_pi<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>(time_std);
}
