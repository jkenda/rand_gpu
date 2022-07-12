#include <iostream>
#include <random>
#include "../include/RNG.hpp"
#include "Timer.hpp"

#define NTHREADS 16
#define ITER (1000000000UL)

using namespace std;

float time_s()
{
    float time;
    {
        TIMER(time);
        #pragma omp parallel num_threads(NTHREADS)
        {
            long num;
            random_device rd;
            mt19937_64 generator(rd());

            for (uint_fast64_t i = 0; i < ITER; i++) {
                float a = generator();
                float b = generator();
            }
        }
    }
    return time;
}

float time_l(size_t multi)
{
    float time;
    {
        TIMER(time);
        #pragma omp parallel num_threads(NTHREADS)
        {
            long num;
            rand_gpu::RNG rng(multi);

            for (uint_fast64_t i = 0; i < ITER; i++) {
                float a = rng.get_random();
                float b = rng.get_random();
            }
        }
    }
    return time;
}

int main()
{
    float time_std = time_s();

    for (int i = 1; i <= 128; i *= 2) {

    }
}
