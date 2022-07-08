#include "../src/RNG.hpp"
#include <iostream>
#include <cstdint>

using namespace std;

int main()
{
    for (int i = RAND_GPU_ALGORITHM_KISS09; i <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; i++)
    {
        rand_gpu_algorithm algorithm = static_cast<rand_gpu_algorithm>(i);
        rand_gpu::RNG rng(2, 1, algorithm, 0);
        cout << rand_gpu::algorithm_name(algorithm) << ":\n";
        cout << "init time: " << rng.init_time() << '\n';

        for (size_t j = 0; j < 256; j++)
        {
            cout << std::to_string(rng.get_random<uint8_t>()) << ' ';
        }
        cout << "\n\n";
    }
}
