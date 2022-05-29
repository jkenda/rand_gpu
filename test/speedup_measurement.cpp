#include "../src/RNG.hpp"
#include <chrono>
#include <random>
#include <iostream>

using namespace std;
using chrono::duration;
using chrono::seconds;
using chrono::microseconds;
using chrono::system_clock;
using chrono::duration_cast;

mt19937 generator(duration_cast<microseconds>(system_clock::now().time_since_epoch()).count());


uint64_t num_generated_cpu(const seconds duration, const uint32_t percent_calc)
{
    const uint32_t percent_gen = 100 - percent_calc;
    uint64_t num_generated = 0;

    auto start = system_clock::now();

    while (system_clock::now() - start < duration)
    {
        auto start_gen = system_clock::now();
        generator();
        auto duration_gen = system_clock::now() - start_gen;
        num_generated++;

        auto duration_calc = duration_gen * percent_calc / percent_gen;
        auto start_calc = system_clock::now();
        while (system_clock::now() - start_calc < duration_calc);
    }

    return num_generated;
}

uint64_t num_generated_gpu(const seconds duration, const uint32_t percent_calc, rand_gpu::RNG &rng)
{
    const uint32_t percent_gen = 100 - percent_calc;
    uint64_t num_generated = 0;

    auto start = system_clock::now();

    while (system_clock::now() - start < duration)
    {
        auto start_gen = system_clock::now();
        rng.get_random<uint32_t>();
        auto duration_gen = system_clock::now() - start_gen;
        num_generated++;

        auto duration_calc = duration_gen * percent_calc / percent_gen;
        auto start_calc = system_clock::now();
        while (system_clock::now() - start_calc < duration_calc);
    }

    return num_generated;
}

int main()
{
    cout << "multi;percent;speedup\n";
    for (uint32_t multi = 1; multi <= 64; multi *= 2)
    {
        rand_gpu::RNG rng(multi);

        for (uint32_t percent_calc = 0; percent_calc <= 90; percent_calc += 10)
        {
            uint64_t num_cpu = num_generated_cpu(10s, percent_calc);
            uint64_t num_gpu = num_generated_gpu(10s, percent_calc, rng);
            cout << multi << ';' << percent_calc << ';' << (num_gpu / (float) num_cpu) << '\n';
        }
    }
}
