#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <omp.h>
#include "../include/RNG.hpp"

#define SAMPLES (10000000000UL)

using namespace std;
using std::chrono::system_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

float pi_std(int nthreads)
{
    uint64_t count = 0;
    random_device rd;
    mt19937 seed(rd());
    omp_lock_t lock;
    omp_init_lock(&lock);

    #pragma omp parallel num_threads(nthreads)
    {
        uint_fast64_t cnt_l = 0;
        omp_set_lock(&lock);
        mt19937 generator(seed());
        omp_unset_lock(&lock);

        for (unsigned long i = 0; i < SAMPLES / nthreads; i++) {
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

float pi_lib(size_t n_buffers, size_t multi, int nthreads)
{
    uint64_t count = 0;

    #pragma omp parallel num_threads(nthreads)
    {
        rand_gpu::RNG rng(n_buffers, multi);

        uint_fast64_t cnt_l = 0;

        for (unsigned long i = 0; i < SAMPLES / nthreads; i++) {
            float a = rng.get_random<float>();
            float b = rng.get_random<float>();
            if (sqrt(a*a + b*b) < 1.0f)
                cnt_l++;
        }

        #pragma omp atomic
        count += cnt_l;

        cout << omp_get_thread_num() << ": " << rng.buffer_misses() << " misses\n";
    }

    return (float) count / SAMPLES * 4;
}

int main(int argc, char **argv)
{
    size_t n_buffers = 4;
    size_t multi = 16;
    int nthreads = omp_get_max_threads();
    if (argc >= 2)
        sscanf(argv[1], "%lu", &n_buffers);
    if (argc >= 3)
        sscanf(argv[2], "%lu", &multi);
    if (argc == 4)
        sscanf(argv[3], "%d", &nthreads);
    
    cout << "num. buffers: " << n_buffers << ", multi: " << multi << ", nthreads: " << nthreads << '\n';
    cout << "real pi: " << M_PI << '\n';

    auto start = system_clock::now();
    float pi = pi_lib(n_buffers, multi, nthreads);
    float time_lib = duration_cast<microseconds>(system_clock::now() - start).count() / (float) 1'000'000;
    cout << "lib pi ≃ " << pi << " (+-" << abs(pi - M_PI) << "), " << time_lib << " s\n";

    start = system_clock::now();
    pi = pi_std(nthreads);
    float time_std = duration_cast<microseconds>(system_clock::now() - start).count() / (float) 1'000'000;
    cout << "std pi ≃ " << pi << " (+-" << abs(pi - M_PI) << "), " << time_std << " s\n";

    cout << "speedup = " << time_std / (float) time_lib << '\n';
}
