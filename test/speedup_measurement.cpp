#include "../src/RNG.hpp"
#include <chrono>
#include <random>
#include <iostream>
#include <cstdlib>
#include <ctime>

#pragma GCC diagnostic ignored "-Wunused-variable"
#define SAMPLES 10

using namespace std;
using chrono::duration;
using chrono::seconds;
using chrono::microseconds;
using chrono::system_clock;
using chrono::high_resolution_clock;
using chrono::duration_cast;

mt19937 generator(duration_cast<microseconds>(system_clock::now().time_since_epoch()).count());


uint64_t num_generated_cpu_c(const seconds duration, const uint32_t percent_calc)
{
    const uint32_t percent_gen = 100 - percent_calc;
    size_t num_generated = 0;

    auto start = high_resolution_clock::now();

    while (high_resolution_clock::now() - start < duration)
    {
        auto start_gen = high_resolution_clock::now();
        for (int i = 0; i < SAMPLES; i++)
            volatile float a = rand() / (float) RAND_MAX;
        auto duration_gen = high_resolution_clock::now() - start_gen;

        auto start_calc = high_resolution_clock::now();
        auto finish_calc = start_calc + duration_gen * percent_calc / percent_gen;
        while (high_resolution_clock::now() < finish_calc);
        num_generated++;
    }

    return num_generated;
}

uint64_t num_generated_cpu_cpp(const seconds duration, const uint32_t percent_calc)
{
    const uint32_t percent_gen = 100 - percent_calc;
    size_t num_generated = 0;

    auto start = high_resolution_clock::now();

    while (high_resolution_clock::now() - start < duration)
    {
        auto start_gen = high_resolution_clock::now();
        for (int i = 0; i < SAMPLES; i++)
            volatile float a = generator() / (float) UINT32_MAX;
        auto duration_gen = high_resolution_clock::now() - start_gen;

        auto start_calc = high_resolution_clock::now();
        auto finish_calc = start_calc + duration_gen * percent_calc / percent_gen;
        while (high_resolution_clock::now() < finish_calc);
        num_generated++;
    }

    return num_generated;
}

uint64_t num_generated_gpu(const seconds duration, const uint32_t percent_calc, rand_gpu::RNG &rng)
{
    const uint32_t percent_gen = 100 - percent_calc;
    size_t num_generated = 0;

    auto start = high_resolution_clock::now();

    while (high_resolution_clock::now() - start < duration)
    {
        auto start_gen = high_resolution_clock::now();
        for (int i = 0; i < SAMPLES; i++)
            volatile float a = rng.get_random<float>();
        auto duration_gen = high_resolution_clock::now() - start_gen;

        auto start_calc = high_resolution_clock::now();
        auto finish_calc = start_calc + duration_gen * percent_calc / percent_gen;
        while (high_resolution_clock::now() < finish_calc);
        num_generated++;
    }

    return num_generated;
}

int main()
{
    int num_cpu_c[10];
    int num_cpu_cpp[10];

    cout << "measuring CPU times ...\n";
    for (uint32_t percent_calc = 0; percent_calc <= 90; percent_calc += 10)
    {
        cout << "\r" << percent_calc << " %"; flush(cout);
        num_cpu_c[percent_calc/10] = num_generated_cpu_c(2s, percent_calc);
        num_cpu_cpp[percent_calc/10] = num_generated_cpu_cpp(2s, percent_calc);
    }
    cout << "\r100 %\n";

    cout << "measuring GPU times...\n";
    cout << "n_buffers;multi;percent;speedup C;speedup C++\n";

    for (size_t n_buffers = 2; n_buffers <= 64; n_buffers *= 2)
    {
        for (size_t multi = 1; multi <= 64; multi *= 2)
        {
            srand(time(NULL));
            rand_gpu::RNG rng(n_buffers, multi);

            for (uint32_t percent_calc = 0; percent_calc <= 90; percent_calc += 10)
            {
                uint64_t num_gpu = num_generated_gpu(2s, percent_calc, rng);
                cout << n_buffers << ';' << multi << ';' << percent_calc << ';' << (num_gpu / (float) num_cpu_c[percent_calc/10]) << ';' << (num_gpu / (float) num_cpu_cpp[percent_calc/10]) << '\n';
            }
        }
    }
}
