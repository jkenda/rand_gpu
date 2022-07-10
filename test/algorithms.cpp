#include "../src/RNG.hpp"
#include <iostream>
#include <cstdint>
#include <cmath>
#include <chrono>

#define SAMPLES (500000000UL)

using std::chrono::system_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

using namespace std;

int main()
{
    for (int i = RAND_GPU_ALGORITHM_KISS09; i <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; i++)
    {
        rand_gpu_algorithm algorithm = static_cast<rand_gpu_algorithm>(i);
        rand_gpu::RNG rng(0, algorithm, 2, 1);
        cout << rand_gpu::algorithm_name(algorithm, true) << ":\n";
        cout << "init time: " << rng.init_time() << " ms\n";

        for (size_t j = 0; j < 256; j++)
        {
            cout << std::to_string(rng.get_random<uint8_t>()) << ' ';
        }
        cout << "\n\n";
    }


    long cnt = 0;

    srand(time(NULL));

    auto start = system_clock::now();

    for (unsigned long i = 0; i < SAMPLES; i++) {
        float a = (float) rand() / RAND_MAX;
        float b = (float) rand() / RAND_MAX;
        if (a*a + b*b < 1.0f) {
            cnt++;
        }
    }
    double pi = (double) cnt / SAMPLES * 4;
    cout << "pi ≃ " << pi << " (+-" << abs(pi - M_PI) << ")\n";

    float time_std = duration_cast<microseconds>(system_clock::now() - start).count() / (float) 1'000'000;

    for (int i = RAND_GPU_ALGORITHM_KISS09; i <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; i++)
    {
        rand_gpu_algorithm algorithm = static_cast<rand_gpu_algorithm>(i);
        rand_gpu::RNG rng(0, algorithm, 8, 32);

        start = system_clock::now();
            
        long cnt = 0;

        for (unsigned long i = 0; i < SAMPLES; i++) {
            float a = rng.get_random<float>();
            float b = rng.get_random<float>();
            if (a*a + b*b < 1.0f) {
                cnt++;
            }
        }
        pi = cnt / (double) SAMPLES * 4;

        float time_lib = duration_cast<microseconds>(system_clock::now() - start).count() / (float) 1'000'000;

        cout << "pi ≃ " << pi << " (+-" << abs(pi - M_PI) << "), ";
        cout << "speedup = " << time_std / time_lib << ", ";
        cout << rng.buffer_misses() << " misses - ";
        cout << rand_gpu::algorithm_name(algorithm) << '\n';
    }
}
