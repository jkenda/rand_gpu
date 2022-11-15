#include <iostream>
#include <cstdint>
#include "../include/RNG.hpp"

#define SAMPLES (256*1024)

using namespace std;

size_t frequency[256];

int main()
{
    rand_gpu::RNG<RAND_GPU_ALGORITHM_MT19937> rng(2, 1);

    for (size_t i = 0; i < SAMPLES; i++)
    {
        auto num = rng.get_random<uint8_t>();
        frequency[num]++;
    }

    cout << "number;frequency\n";

    for (int i = 0; i < 256; i++)
    {
        cout << i << ';' << frequency[i] << '\n';
    }
}
