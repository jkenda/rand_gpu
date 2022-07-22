#include "../include/RNG.hpp"
#include <chrono>
#include <random>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#pragma GCC diagnostic ignored "-Wunused-variable"
#define SAMPLES 10
#define C 0
#define CPP 1
#define FLT 0
#define DBL 1

struct parameters
{
    rand_gpu_algorithm algorithm;
    int n_buffers, multi;
    float speedup;
};

using namespace std;
using chrono::duration;
using chrono::seconds;
using chrono::nanoseconds;
using chrono::system_clock;
using chrono::high_resolution_clock;
using chrono::duration_cast;

static const int N_ALGORITHMS = RAND_GPU_ALGORITHM_XORSHIFT6432STAR - RAND_GPU_ALGORITHM_KISS09 + 1;
static const duration DURATION = 2s;


mt19937 generator32(system_clock::now().time_since_epoch().count());
mt19937_64 generator64(system_clock::now().time_since_epoch().count());


template <typename T>
uint64_t num_generated_cpu_c(const nanoseconds duration, const uint32_t percent_calc)
{
    const uint32_t percent_gen = 100 - percent_calc;
    size_t num_generated = 0;

    auto start = high_resolution_clock::now();

    while (high_resolution_clock::now() - start < duration)
    {
        auto start_gen = high_resolution_clock::now();
        for (int i = 0; i < SAMPLES; i++)
            volatile T a = rand() / (T) RAND_MAX;
        auto duration_gen = high_resolution_clock::now() - start_gen;

        auto start_calc = high_resolution_clock::now();
        auto finish_calc = start_calc + duration_gen * percent_calc / percent_gen;
        while (high_resolution_clock::now() < finish_calc);
        num_generated++;
    }

    return num_generated;
}

template <typename T>
uint64_t num_generated_cpu_cpp(const nanoseconds duration, const uint32_t percent_calc)
{
    const uint32_t percent_gen = 100 - percent_calc;
    size_t num_generated = 0;

    auto start = high_resolution_clock::now();

    while (high_resolution_clock::now() - start < duration)
    {
        auto start_gen = high_resolution_clock::now();
        for (int i = 0; i < SAMPLES; i++)
            volatile T a = (is_same<T, double>::value)
                ? generator64() / (T) UINT64_MAX
                : generator32() / (T) UINT32_MAX;
        auto duration_gen = high_resolution_clock::now() - start_gen;

        auto start_calc = high_resolution_clock::now();
        auto finish_calc = start_calc + duration_gen * percent_calc / percent_gen;
        while (high_resolution_clock::now() < finish_calc);
        num_generated++;
    }

    return num_generated;
}

template <typename T>
uint64_t num_generated_gpu(const nanoseconds duration, const uint32_t percent_calc, rand_gpu_rng *rng)
{
    const uint32_t percent_gen = 100 - percent_calc;
    size_t num_generated = 0;

    auto start = high_resolution_clock::now();

    while (high_resolution_clock::now() - start < duration)
    {
        auto start_gen = high_resolution_clock::now();
        for (int i = 0; i < SAMPLES; i++)
            volatile T a = is_same<T, double>::value ? rand_gpu_double(rng) : rand_gpu_float(rng);
        auto duration_gen = high_resolution_clock::now() - start_gen;

        auto start_calc = high_resolution_clock::now();
        auto finish_calc = start_calc + duration_gen * percent_calc / percent_gen;
        while (high_resolution_clock::now() < finish_calc);
        num_generated++;
    }

    return num_generated;
}

int main(int argc, char **argv)
{
    uint lang = C, type = FLT;
    if (argc >= 2 && strcmp(argv[1], "C++"))
        lang = CPP;
    if (argc >= 3 && strcmp(argv[2], "double"))
        type = DBL;

    int num_cpu[10];

    // algorithm, n_buffers, multi, %
    float speedup[N_ALGORITHMS][4][7][10];

    srand(time(NULL));

    cout << "measuring CPU times ...\n";

    cout << "percent,num_generated\n";
    for (int i = 0; i < 10; i++)
    {
        int percent_calc = i * 10;
        if (lang == C) {
            num_cpu[i] = (type == FLT)
                ? num_generated_cpu_c<float>(DURATION, percent_calc)
                : num_generated_cpu_c<double>(DURATION, percent_calc);
        }
        else {
            num_cpu[i] = (type == FLT)
                ? num_generated_cpu_cpp<float>(DURATION, percent_calc)
                : num_generated_cpu_cpp<double>(DURATION, percent_calc);
        }

        printf("%d,%e\n", percent_calc, (double) num_cpu[i]);
    }
    cout << '\n';

    cout << "measuring GPU times...\n";

    for (int algorithm = RAND_GPU_ALGORITHM_KISS09; algorithm <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; algorithm++)
    {
        cout << "algorithm,n_buffers,multi,0 %,10 %,20 %,30 %,40 %,50 %,60 %,70 %,80 %,90 %\n";

        for (int n_buffers = 2; n_buffers <= 16; n_buffers *= 2)
        {
            for (int multi = 1; multi <= 64; multi *= 2)
            {
                printf("%s,%02d,%02d", rand_gpu::algorithm_name((rand_gpu_algorithm) algorithm), n_buffers, multi);
                fflush(stdout);

                rand_gpu_rng *rng = rand_gpu_new((rand_gpu_algorithm) algorithm, n_buffers, multi);
                
                for (int i = 0; i < 10; i++)
                {
                    int percent_calc = i * 10;
                    uint64_t num_gpu = (type == FLT) 
                        ? num_generated_gpu<float>(DURATION, percent_calc, rng)
                        : num_generated_gpu<double>(DURATION, percent_calc, rng);

                    float speedup_current = num_gpu / (float) num_cpu[i];

                    printf(",%.5f", speedup_current);
                    flush(cout);

                    int n_buffers_offset = log2(n_buffers) - 1;
                    int multi_offset = log2(multi);
                    speedup[algorithm][n_buffers_offset][multi_offset][i] = speedup_current;
                }
                cout << '\n';
                rand_gpu_delete(rng);
            }
        }
        cout << '\n';

    }


    // get the best combination of parameters for each get_random/calculation ratio

    parameters best_parameters_percent[10];

    for (int i = 0; i < 10; i++)
    {
        float best_speedup = 0;

        for (int algorithm = RAND_GPU_ALGORITHM_KISS09; algorithm <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; algorithm++)
        {
            for (int n_buffers = 2; n_buffers <= 16; n_buffers *= 2)
            {
                for (int multi = 1; multi <= 64; multi *= 2)
                {
                    int n_buffers_offset = log2(n_buffers) - 1;
                    int multi_offset = log2(multi);
                    float current_speedup = speedup[algorithm][n_buffers_offset][multi_offset][i];
                    if (current_speedup > best_speedup)
                    {
                        best_speedup = current_speedup;
                        best_parameters_percent[i] = 
                            { (rand_gpu_algorithm) algorithm, n_buffers, multi, current_speedup };
                    }
                }
            }
        }
    }

    cout << "Best parameters for each workload type:\n";
    cout << "percentage,algorithm,n_buffers,multi,speedup\n";

    for (int i = 0; i < 10; i++)
    {
        printf("%d,%s,%d,%d,%f\n", i * 10, rand_gpu::algorithm_name(best_parameters_percent[i].algorithm),
            best_parameters_percent[i].n_buffers, best_parameters_percent[i].multi, best_parameters_percent[i].speedup);
    }
    cout << '\n';


    // get the best combination of parameters for each algorithm

    parameters best_parameters_algorithm[N_ALGORITHMS];

    for (int algorithm = RAND_GPU_ALGORITHM_KISS09; algorithm <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; algorithm++)
    {
        float best_speedup = 0;

        for (int n_buffers = 2; n_buffers <= 16; n_buffers *= 2)
        {
            for (int multi = 1; multi <= 64; multi *= 2)
            {
                int n_buffers_offset = log2(n_buffers) - 1;
                int multi_offset = log2(multi);
                float avg_speedup = 0.0f;

                for (int i = 0; i < 10; i++)
                {
                    float current_speedup = speedup[algorithm][n_buffers_offset][multi_offset][i];
                    avg_speedup += current_speedup;
                }
                avg_speedup /= 10;

                if (avg_speedup > best_speedup)
                {
                    best_speedup = avg_speedup;
                    best_parameters_algorithm[algorithm] = 
                        { (rand_gpu_algorithm) algorithm, n_buffers, multi, avg_speedup };
                }
            }
        }
    }

    cout << "Best parameters for algorithm:\n";
    cout << "algorithm,n_buffers,multi,speedup\n";

    for (int i = 0; i < N_ALGORITHMS; i++)
    {
        printf("%s,%d,%d,%f\n", rand_gpu::algorithm_name((rand_gpu_algorithm) i),
            best_parameters_algorithm[i].n_buffers, best_parameters_algorithm[i].multi, best_parameters_algorithm[i].speedup);
    }

}
