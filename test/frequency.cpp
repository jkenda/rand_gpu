#include <iostream>
#include <cstdint>
#include "../include/RNG.hpp"

#define SAMPLES (256*10)

using namespace std;

size_t frequency[256];

int main()
{
    rand_gpu::RNG rng(RAND_GPU_ALGORITHM_TYCHE, 8, 64);

    cout << "counting...";

    for (size_t i = 0; i < SAMPLES; i++)
    {
        auto num = rng.get_random<uint8_t>();
        frequency[num]++;
    }

    cout << "\rnumber;frequency\n";

    for (int i = 0; i < 256; i++)
    {
        cout << i << ';' << frequency[i] << '\n';
    }
}
