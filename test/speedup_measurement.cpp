#include "../include/RNG.hpp"
#include <chrono>
#include <cstdint>
#include <random>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <array>

#define C 0
#define CPP 1
#define FLT 0
#define DBL 1

#pragma GCC diagnostic ignored "-Wunused-variable"
#define SAMPLES 5

struct parameters
{
    const char *algorithm_name;
    int n_buffers, multi;
    float speedup;
};

struct algorithm_speedup
{
    const char *algorithm_name;
    float speedup;
    bool operator<(const algorithm_speedup &other) { return other.speedup < speedup; }
};

using namespace std;
using chrono::duration;
using chrono::seconds;
using chrono::nanoseconds;
using chrono::steady_clock;

static const int N_ALGORITHMS = RAND_GPU_ALGORITHM_XORSHIFT6432STAR - RAND_GPU_ALGORITHM_KISS09 + 1;
static const duration<uint64_t> DURATION(1s);
static const array<int, 12> multipliers { 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64 };
static const array<int, 6> n_bufs { 2, 3, 4, 6, 8, 12 };


mt19937 generator32(steady_clock::now().time_since_epoch().count());
mt19937_64 generator64(steady_clock::now().time_since_epoch().count());

nanoseconds time_calc[11];

template <typename T>
uint64_t num_generated_cpu_c(const nanoseconds duration, const uint32_t percent_gen)
{
    const uint32_t percent_calc = 100 - percent_gen;
    const uint32_t i = percent_gen / 10;
    size_t num_generated = 0;

    auto start = steady_clock::now();

    if (percent_calc == 100) {
        while (steady_clock::now() - start < duration)
            num_generated++;
        return num_generated;
    }

    while (steady_clock::now() - start < duration)
    {
        auto start_gen = steady_clock::now();
        for (int i = 0; i < SAMPLES; i++)
            volatile T a = rand() / (T) RAND_MAX;
        auto duration_gen = chrono::steady_clock::now() - start_gen;
        auto duration_calc = duration_gen * percent_calc / percent_gen;

        auto start_calc = chrono::steady_clock::now();
        auto finish_calc = start_calc + duration_calc;
        while (steady_clock::now() < finish_calc);
        num_generated++;
        time_calc[i] += duration_calc;
    }
    time_calc[i] /= num_generated;

    return num_generated;
}

template <typename T>
uint64_t num_generated_cpu_cpp(const nanoseconds duration, const uint32_t percent_gen)
{
    const uint32_t percent_calc = 100 - percent_gen;
    const uint32_t i = percent_gen / 10;
    size_t num_generated = 0;

    auto start = steady_clock::now();

    if (percent_calc == 100) {
        while (steady_clock::now() - start < duration)
            num_generated++;
        return num_generated;
    }
    
    while (steady_clock::now() - start < duration)
    {
        auto start_gen = steady_clock::now();
        for (int i = 0; i < SAMPLES; i++)
            volatile T a = (is_same<T, double>::value)
                ? generator64() / (T) UINT64_MAX
                : generator32() / (T) UINT32_MAX;
        auto duration_gen = steady_clock::now() - start_gen;
        auto duration_calc = duration_gen * percent_calc / percent_gen;

        auto start_calc = steady_clock::now();
        auto finish_calc = start_calc + duration_calc;
        while (steady_clock::now() < finish_calc);
        num_generated++;
        time_calc[i] += duration_calc;
    }
    time_calc[i] /= num_generated;

    return num_generated;
}

template <typename T>
uint64_t num_generated_gpu(const nanoseconds duration, const uint32_t percent_gen, rand_gpu_rng *rng)
{
    const uint32_t percent_calc = 100 - percent_gen;
    const uint32_t i = percent_gen / 10;
    size_t num_generated = 0;

    auto start = steady_clock::now();

    if (percent_calc == 100) {
        while (steady_clock::now() - start < duration)
            num_generated++;
        return num_generated;
    }
    
    while (steady_clock::now() - start < duration)
    {
        for (int i = 0; i < SAMPLES; i++)
            volatile T a = is_same<T, double>::value ? rand_gpu_double(rng) : rand_gpu_float(rng);

        auto start_calc = steady_clock::now();
        auto finish_calc = start_calc + time_calc[i];
        while (steady_clock::now() < finish_calc);
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

    int num_cpu[11];

    // algorithm, n_buffers, multi, %
    float speedup[N_ALGORITHMS][4][7][11];

    // compilation and initilization times

    cout << "Compilation time for each algorithm:\n\n";
    cout << "algorithm,compilation time [ms]\n";

    for (int algorithm = RAND_GPU_ALGORITHM_KISS09; algorithm <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; algorithm++)
    {
        rand_gpu_rng *rng = rand_gpu_new_rng((rand_gpu_algorithm) algorithm, 2, 1);
        rand_gpu_delete_all();
        printf("%s,%f\n", rand_gpu::algorithm_name((rand_gpu_algorithm) algorithm), 
            rand_gpu_compilation_time((rand_gpu_algorithm) algorithm));
    }
    cout << '\n';

    cout << "Initilization times for each algorithm [ms]:\n\n";

    for (int algorithm = RAND_GPU_ALGORITHM_KISS09; algorithm <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; algorithm++)
    {
        cout << "algorithm,multi";
        for (int n_buffers : n_bufs) cout << ',' << n_buffers << " buffers";
        cout << '\n';

        for (int multi : multipliers)
        {
            printf("%s,%02d", rand_gpu::algorithm_name((rand_gpu_algorithm) algorithm), multi);
            for (int n_buffers : n_bufs)
            {
                rand_gpu_rng *rng = rand_gpu_new_rng((rand_gpu_algorithm) algorithm, n_buffers, multi);
                printf(",%.5f", rand_gpu_rng_init_time(rng));
                rand_gpu_delete_all();
            }
            cout << endl;
        }
        cout << endl;
    }


    srand(time(NULL));

    cout << "measuring CPU times ...\n\n";

    cout << "percent,num_generated\n";
    for (int i = 0; i <= 10; i++)
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
        flush(cout);
    }
    cout << '\n';

    cout << "measuring GPU times for different proportion of generation/calculation (0 % generation should always be around 1)...\n\n";

    for (int algorithm = RAND_GPU_ALGORITHM_KISS09; algorithm <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; algorithm++)
    {
        cout << "algorithm,n_buffers,multi,0 %,10 %,20 %,30 %,40 %,50 %,60 %,70 %,80 %,90 %,100 %\n";

        for (int n_buffers : n_bufs)
        {
            for (int multi : multipliers)
            {
                printf("%s,%02d,%02d", rand_gpu::algorithm_name((rand_gpu_algorithm) algorithm), n_buffers, multi);
                fflush(stdout);

                rand_gpu_rng *rng = rand_gpu_new_rng((rand_gpu_algorithm) algorithm, n_buffers, multi);
                
                for (int i = 0; i <= 10; i++)
                {
                    int percent_gen = i * 10;
                    uint64_t num_gpu = (type == FLT) 
                        ? num_generated_gpu<float>(DURATION, percent_gen, rng)
                        : num_generated_gpu<double>(DURATION, percent_gen, rng);

                    float speedup_current = num_gpu / (float) num_cpu[i];

                    int n_buffers_offset = log2(n_buffers) - 1;
                    int multi_offset = log2(multi);
                    speedup[algorithm][n_buffers_offset][multi_offset][i] = speedup_current;

                    printf(",%7f", speedup_current);
                    flush(cout);
                }
                cout << '\n';
                rand_gpu_delete_rng(rng);
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
            for (int n_buffers : n_bufs)
            {
                for (int multi : multipliers)
                {
                    int n_buffers_offset = log2(n_buffers) - 1;
                    int multi_offset = log2(multi);
                    float current_speedup = speedup[algorithm][n_buffers_offset][multi_offset][i];
                    if (current_speedup > best_speedup)
                    {
                        best_speedup = current_speedup;
                        best_parameters_percent[i] = 
                            { rand_gpu::algorithm_name((rand_gpu_algorithm) algorithm), n_buffers, multi, current_speedup };
                    }
                }
            }
        }
    }

    cout << "Best parameters for each workload type:\n";
    cout << "percent calc,algorithm,n_buffers,multi,speedup\n";

    for (int i = 0; i < 10; i++)
    {
        printf("%d,%s,%d,%d,%f\n", i * 10, best_parameters_percent[i].algorithm_name,
            best_parameters_percent[i].n_buffers, best_parameters_percent[i].multi, best_parameters_percent[i].speedup);
    }
    cout << '\n';


    // get the best combination of parameters for each algorithm

    parameters best_parameters_algorithm[N_ALGORITHMS];

    for (int algorithm = RAND_GPU_ALGORITHM_KISS09; algorithm <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; algorithm++)
    {
        float best_speedup = 0;

        for (int n_buffers : n_bufs)
        {
            for (int multi : multipliers)
            {
                int n_buffers_offset = log2(n_buffers) - 1;
                int multi_offset = log2(multi);
                float avg_speedup = 0.0f;

                for (int i = 0; i <= 10; i++)
                {
                    float current_speedup = speedup[algorithm][n_buffers_offset][multi_offset][i];
                    avg_speedup += current_speedup;
                }
                avg_speedup /= 10;

                if (avg_speedup > best_speedup)
                {
                    best_speedup = avg_speedup;
                    best_parameters_algorithm[algorithm] = 
                        { rand_gpu::algorithm_name((rand_gpu_algorithm) algorithm), n_buffers, multi, avg_speedup };
                }
            }
        }
    }

    cout << "Best parameters for algorithm:\n";
    cout << "algorithm,n_buffers,multi,speedup\n";

    for (int a = 0; a < N_ALGORITHMS; a++)
    {
        printf("%s,%d,%d,%f\n", rand_gpu::algorithm_name((rand_gpu_algorithm) a),
            best_parameters_algorithm[a].n_buffers, best_parameters_algorithm[a].multi, best_parameters_algorithm[a].speedup);
    }
    cout << '\n';


    // print performance at multi in relation to n_buffers

    cout << "Algorithm performance at different n_buffers in relation to multi (50 % generation):\n\n";

    for (int algorithm = RAND_GPU_ALGORITHM_KISS09; algorithm <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; algorithm++)
    {
        cout << "algorithm,multi";
        for (int n_buffers : n_bufs) cout << ',' << n_buffers << " buffers";
        cout << '\n';

        for (int multi : multipliers)
        {
            int multi_offset = log2(multi);
            printf("%s,%02d", rand_gpu::algorithm_name((rand_gpu_algorithm) algorithm), multi);
            for (int n_buffers : n_bufs)
            {
                int n_buffers_offset = log2(n_buffers) - 1;
                float speedup_current = speedup[algorithm][n_buffers_offset][multi_offset][5];
                printf(",%.5f", speedup_current);
            }
            cout << '\n';
        }
        cout << '\n';
    }

    cout << "Algorithm performance for each generation/calculation ratio:\n\n";

    for (int i = 0; i <= 10; i++)
    {
        algorithm_speedup speedup_algorithm[N_ALGORITHMS];
        int percent_calc = i * 10;
        for (int a = RAND_GPU_ALGORITHM_KISS09; a <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; a++)
        {
            const int n_buffers_offset = log2(2) - 1, multi_offset = log2(64);
            speedup_algorithm[a].algorithm_name = rand_gpu::algorithm_name((rand_gpu_algorithm) a);
            speedup_algorithm[a].speedup = speedup[a][n_buffers_offset][multi_offset][i];
        }

        sort(speedup_algorithm, speedup_algorithm+N_ALGORITHMS);

        cout << "algorithm," << percent_calc << " %\n";
        for (int a = RAND_GPU_ALGORITHM_KISS09; a <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; a++)
        {
            printf("%s,%.5f\n", speedup_algorithm[a].algorithm_name, speedup_algorithm[a].speedup);
        }
        cout << '\n';
    }

}
