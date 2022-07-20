#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <mutex>
#include "../include/RNG.hpp"

#define SAMPLES (10000000000UL)

using namespace std;
using std::chrono::system_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

float pi_std()
{
    uint64_t count = 0;
    random_device rd;
    mt19937 seed(rd());
    mutex lock;

    #pragma omp parallel
    {
        uint_fast64_t cnt_l = 0;
        lock.lock();
        mt19937 generator(seed());
        lock.unlock();

        #pragma omp for schedule(static)
        for (unsigned long i = 0; i < SAMPLES; i++) {
            float a = generator() / (float) UINT32_MAX;
            float b = generator() / (float) UINT32_MAX;
            if (sqrt(a*a + b*b) < 1.0f)
                cnt_l++;
        }

        #pragma omp atomic
        count += cnt_l;
    }

    return (float) count / SAMPLES * 4;
}

float pi_lib(size_t n_buffers, size_t multi)
{
    uint64_t count = 0;

    #pragma omp parallel
    {
        rand_gpu::RNG rng(n_buffers, multi);

        uint_fast64_t cnt_l = 0;

        #pragma omp for schedule(static)
        for (unsigned long i = 0; i < SAMPLES; i++) {
            float a = rng.get_random<float>();
            float b = rng.get_random<float>();
            if (sqrt(a*a + b*b) < 1.0f)
                cnt_l++;
        }

        #pragma omp atomic
        count += cnt_l;

        cout << rng.buffer_misses() << " misses\n";
    }

    return (float) count / SAMPLES * 4;
}

int main(int argc, char **argv)
{
    size_t n_buffers = 4;
    size_t multi = 16;
    if (argc >= 2)
        sscanf(argv[1], "%lu", &n_buffers);
    if (argc >= 3)
        sscanf(argv[2], "%lu", &multi);
    
    cout << "num. buffers: " << n_buffers << ", multi: " << multi << '\n';
    cout << "real pi: " << M_PI << '\n';

    auto start = system_clock::now();
    float pi = pi_lib(n_buffers, multi);
    float time_lib = duration_cast<microseconds>(system_clock::now() - start).count() / (float) 1'000'000;
    cout << "lib pi ≃ " << pi << " (+-" << abs(pi - M_PI) << "), " << time_lib << " s\n";

    start = system_clock::now();
    pi = pi_std();
    float time_std = duration_cast<microseconds>(system_clock::now() - start).count() / (float) 1'000'000;
    cout << "std pi ≃ " << pi << " (+-" << abs(pi - M_PI) << "), " << time_std << " s\n";

    cout << "speedup = " << time_std / (float) time_lib << '\n';
}
