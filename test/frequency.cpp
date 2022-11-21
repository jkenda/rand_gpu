#include <iostream>
#include <cstdint>
#include <array>
#include "../include/RNG.hpp"

typedef uint8_t type;
#define SIZE (1 << (8 * sizeof(type)))
#define SAMPLES (SIZE * 1024 * 128)

using namespace std;

array<size_t, SIZE> frequency;
array<type, SAMPLES> samples = {0};
 
int main()
{
    rand_gpu::RNG<RAND_GPU_ALGORITHM_LFIB> rng(2, 1);
    rng.put_random(&samples, sizeof(samples));

    for (const type &num : samples)
        frequency[num]++;

    cout << "frequency\n";

    for (size_t i = 0; i < SIZE; i++)
        cout << frequency[i] << '\n';
}
