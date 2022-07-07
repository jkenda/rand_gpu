#include <iostream>
#include <unordered_map>
#include <cstdint>
#include <mutex>
#include <omp.h>
#include "../src/RNG.hpp"

#define SAMPLES (1'000'000'000)

using namespace std;

unordered_map<uint32_t, size_t> frequency;
mutex merge_lock;

int main()
{
    frequency.reserve(UINT16_MAX);

    #pragma omp parallel
    {
        unordered_map<uint32_t, size_t> frequency_l;
        frequency_l.reserve(UINT16_MAX);

        rand_gpu::RNG rng(8, 256);

        #pragma omp single
        cout << "counting...";

        #pragma omp for
        for (size_t i = 0; i < SAMPLES; i++)
        {
            auto num = rng.get_random<uint32_t>();
            frequency_l[num]++;
        }

        #pragma omp single
        cout << "\rmerging...";

        {
            lock_guard<mutex> lock(merge_lock);
            frequency.merge(frequency_l);
        }
    }

    cout << "\rnumber;frequency\n";

    for (auto &[num, freq] : frequency)
    {
        cout << to_string(num) << ';' << freq << '\n';
    }
}