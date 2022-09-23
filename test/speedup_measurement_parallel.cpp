#include "../include/RNG.hpp"
#include <chrono>
#include <random>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <mutex>
#include <vector>
#include <array>
#include <omp.h>

#define C 0
#define CPP 1
#define FLT 0
#define DBL 1

#pragma GCC diagnostic ignored "-Wunused-variable"
#define SAMPLES 10

struct parameters
{
    const char *algorithm_name;
    int nbuffers, multi;
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
using chrono::system_clock;
using chrono::high_resolution_clock;
using chrono::duration_cast;

#define N_ALGORITHMS (RAND_GPU_ALGORITHM_XORSHIFT6432STAR - RAND_GPU_ALGORITHM_KISS09 + 1)
#define DURATION 50ms

#define MAX_BUFFERS 16
#define MAX_MULTI   64

#define BUFFER_SAMPLES 4
#define MULTI_SAMPLES  7


mt19937 generator32(system_clock::now().time_since_epoch().count());
mt19937_64 generator64(system_clock::now().time_since_epoch().count());
mutex seed_lock;

array<nanoseconds, 11> time_calc;

template <typename T>
uint64_t num_generated_cpu_c(const nanoseconds duration, const uint32_t percent_gen)
{
    const uint32_t percent_calc = 100 - percent_gen;
    const uint32_t i = percent_gen / 10;
    size_t num_generated = 0;

    seed_lock.lock();
    uint32_t seed = rand();
    seed_lock.unlock();

    auto start = high_resolution_clock::now();

    if (percent_calc == 100) {
        while (high_resolution_clock::now() - start < duration)
            num_generated++;
        return num_generated;
    }

    while (high_resolution_clock::now() - start < duration)
    {
        auto start_gen = high_resolution_clock::now();
        for (int i = 0; i < SAMPLES; i++)
            volatile T a = rand_r(&seed) / (T) RAND_MAX;
        auto duration_gen = high_resolution_clock::now() - start_gen;
        auto duration_calc = duration_gen * percent_calc / percent_gen;

        auto start_calc = high_resolution_clock::now();
        auto finish_calc = start_calc + duration_calc;
        while (high_resolution_clock::now() < finish_calc);
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

    auto start = high_resolution_clock::now();

    if (percent_calc == 100) {
        while (high_resolution_clock::now() - start < duration)
            num_generated++;
        return num_generated;
    }
    
    while (high_resolution_clock::now() - start < duration)
    {
        auto start_gen = high_resolution_clock::now();
        for (int i = 0; i < SAMPLES; i++)
            volatile T a = (is_same<T, double>::value)
                ? generator64() / (T) UINT64_MAX
                : generator32() / (T) UINT32_MAX;
        auto duration_gen = high_resolution_clock::now() - start_gen;
        auto duration_calc = duration_gen * percent_calc / percent_gen;

        auto start_calc = high_resolution_clock::now();
        auto finish_calc = start_calc + duration_calc;
        while (high_resolution_clock::now() < finish_calc);
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

    auto start = high_resolution_clock::now();

    if (percent_calc == 100) {
        while (high_resolution_clock::now() - start < duration)
            num_generated++;
        return num_generated;
    }
    
    while (high_resolution_clock::now() - start < duration)
    {
        for (int i = 0; i < SAMPLES; i++)
            volatile T a = is_same<T, double>::value ? rand_gpu_double(rng) : rand_gpu_float(rng);

        auto start_calc = high_resolution_clock::now();
        auto finish_calc = start_calc + time_calc[i];
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

    int max_threads = omp_get_max_threads();
    int thread_samples = log2(max_threads);

    vector<int> num_cpu(thread_samples);

    // algorithm, n. buffers, multi, n. threads
    //float speedup[N_ALGORITHMS][3][6][max_threads] = {0};
    vector<array<array<array<float, MULTI_SAMPLES>, BUFFER_SAMPLES>, N_ALGORITHMS>> speedup(thread_samples);

    srand(time(NULL));

    cout << "measuring CPU times ...\n\n";

    cout << "nthreads,num_generated\n";

    #pragma omp parallel num_threads(max_threads)
    {
        for (int nthreads = 1; nthreads <= max_threads; nthreads *= 2)
        {
            int i = log2(nthreads);

            #pragma omp for schedule(static)
            for (int n = 0; n < nthreads; n++)
            {
                if (lang == C) {
                    #pragma omp atomic
                    num_cpu[i] += (type == FLT)
                        ? num_generated_cpu_c<float>(DURATION, 50)
                        : num_generated_cpu_c<double>(DURATION, 50);
                }
                else {
                    #pragma omp atomic
                    num_cpu[i] += (type == FLT)
                        ? num_generated_cpu_cpp<float>(DURATION, 50)
                        : num_generated_cpu_cpp<double>(DURATION, 50);
                }
            }

            #pragma omp single
            {
                printf("%d,%e\n", nthreads, (double) num_cpu[i]);
                flush(cout);
            }
        }
    }
    cout << '\n' << endl;

    cout << "measuring GPU times for different n. of threads ...\n\n";

    uint64_t num_gpu, t;

    #pragma omp parallel num_threads(max_threads)
    {

        for (int algorithm = RAND_GPU_ALGORITHM_KISS09; algorithm <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; algorithm++)
        {
            #pragma omp barrier
            int j = algorithm;

            #pragma omp single
            {
                cout << "algorithm,nbuffers,multi,1 thread";
                for (int i = 2; i <= max_threads; i *= 2)
                    cout << ',' << i << " threads";
                cout << '\n';
            }
            #pragma omp barrier

            for (int nbuffers = 2; nbuffers <= MAX_BUFFERS; nbuffers *= 2)
            {
                #pragma omp barrier
                int k = log2(nbuffers) - 1;

                for (int multi = 1; multi <= MAX_MULTI; multi *= 2)
                {
                    #pragma omp barrier
                    if (algorithm == RAND_GPU_ALGORITHM_MT19937 && multi >= 64)
                        break;

                    #pragma omp single
                    {
                        printf("%s,%02d,%02d", rand_gpu::algorithm_name((rand_gpu_algorithm) algorithm), nbuffers, multi);
                        fflush(stdout);
                    }

                    int l = log2(multi);
                    rand_gpu_rng *rng = rand_gpu_new((rand_gpu_algorithm) algorithm, nbuffers, multi);

                    for (int nthreads = 1; nthreads <= max_threads; nthreads *= 2)
                    {
                        int i = log2(nthreads);

                        #pragma omp single
                        {
                            num_gpu = 0;
                            t = 0;
                        }

                        #pragma omp for
                        for (int n = 0; n < nthreads; n++)
                        {
                            #pragma omp atomic
                            t++;

                            #pragma omp atomic
                            num_gpu += (type == FLT) 
                                ? num_generated_gpu<float>(DURATION, 50, rng)
                                : num_generated_gpu<double>(DURATION, 50, rng);
                        }

                        #pragma omp single
                        {
                            speedup[i][j][k][l] = num_gpu / (float) num_cpu[i];
                            //printf(",%.5f", speedup[i][j][k][l]);
                            //fflush(stdout);
                            cout << ',' << t;
                        }
                        #pragma omp barrier
                    }

                    #pragma omp single
                    cout << '\n';

                    rand_gpu_delete(rng);

                    #pragma omp barrier
                }
            }

            #pragma omp single
            cout << endl;
        }
        #pragma omp single
        cout << "end" << endl;
    }

    // get the best combination of parameters for each n. of threads

    vector<parameters> best_parameters_thread(thread_samples);

    for (int i = 0; i < thread_samples; i++)
    {
        float best_speedup = 0;

        for (int j = RAND_GPU_ALGORITHM_KISS09; j <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; j++)
        {
            for (int nbuffers = 2; nbuffers <= MAX_BUFFERS; nbuffers *= 2)
            {
                int k = log2(nbuffers) - 1;

                for (int multi = 1; multi <= MAX_BUFFERS; multi *= 2)
                {
                    int l = log2(multi);
                
                    float current_speedup = speedup[i][j][k][l];
                    if (current_speedup > best_speedup)
                    {
                        best_speedup = current_speedup;
                        best_parameters_thread[i] = 
                            { rand_gpu::algorithm_name((rand_gpu_algorithm) j), nbuffers, multi, current_speedup };
                    }
                }
            }
        }
    }

    cout << "Best parameters for each n. of threads:\n";
    cout << "nthreads,algorithm,nbuffers,multi,speedup\n";

    for (int nthreads = 1; nthreads <= max_threads; nthreads *= 2)
    {
        int i = log2(nthreads);
        printf("%d,%s,%d,%d,%f\n", nthreads, best_parameters_thread[i].algorithm_name,
            best_parameters_thread[i].nbuffers, best_parameters_thread[i].multi, best_parameters_thread[i].speedup);
    }
    cout << '\n';


    // get the best combination of parameters for each algorithm

    parameters best_parameters_algorithm[N_ALGORITHMS];

    for (int j = RAND_GPU_ALGORITHM_KISS09; j <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; j++)
    {
        float best_speedup = 0;

        for (int nbuffers = 2; nbuffers <= MAX_BUFFERS; nbuffers *= 2)
        {
            int k = log2(nbuffers) - 1;

            for (int multi = 1; multi <= MAX_MULTI; multi *= 2)
            {
                int l = log2(multi);
                float avg_speedup = 0.0f;

                for (int i = 0; i < thread_samples; i++)
                {
                    float current_speedup = speedup[i][j][k][l];
                    avg_speedup += current_speedup;
                }
                avg_speedup /= max_threads;

                if (avg_speedup > best_speedup)
                {
                    best_speedup = avg_speedup;
                    best_parameters_algorithm[j] = 
                        { rand_gpu::algorithm_name((rand_gpu_algorithm) j), nbuffers, multi, avg_speedup };
                }
            }
        }
    }

    cout << "Best parameters for algorithm:\n";
    cout << "algorithm,nbuffers,multi,speedup\n";

    for (int a = 0; a < N_ALGORITHMS; a++)
    {
        printf("%s,%d,%d,%f\n", rand_gpu::algorithm_name((rand_gpu_algorithm) a),
            best_parameters_algorithm[a].nbuffers, best_parameters_algorithm[a].multi, best_parameters_algorithm[a].speedup);
    }
    cout << '\n';


    // print performance at multi in relation to nbuffers

    cout << "Algorithm performance in relation to n. buffers at different n. nthreads:\n\n";

    for (int algorithm = RAND_GPU_ALGORITHM_KISS09; algorithm <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; algorithm++)
    {
        int j = algorithm;

        cout << "algorithm,nbuffers,1 thread";
        for (int i = 2; i <= max_threads; i *= 2)
            cout << ',' << i << " threads";
        cout << '\n';

        for (int multi = 1; multi <= MAX_MULTI; multi *= 2)
        {
            int k = log2(multi);
            printf("%s,%02d", rand_gpu::algorithm_name((rand_gpu_algorithm) algorithm), multi);

            for (int nthreads = 1; nthreads <= max_threads; nthreads *= 2)
            {
                int i = log2(nthreads);
                int l = log2(MAX_MULTI);
                float speedup_current = speedup[i][j][k][l];
                printf(",%.5f", speedup_current);
            }
            cout << '\n';
        }
        cout << '\n';
    }

    cout << "Algorithm performance in relation to multi at different n. nthreads:\n\n";

    for (int algorithm = RAND_GPU_ALGORITHM_KISS09; algorithm <= RAND_GPU_ALGORITHM_XORSHIFT6432STAR; algorithm++)
    {
        int j = algorithm;

        cout << "algorithm,multi,1 thread";
        for (int i = 2; i <= max_threads; i *= 2)
            cout << ',' << i << " threads";
        cout << '\n';

        for (int multi = 1; multi <= MAX_MULTI; multi *= 2)
        {
            int l = log2(multi);
            printf("%s,%02d", rand_gpu::algorithm_name((rand_gpu_algorithm) algorithm), multi);

            for (int nthreads = 1; nthreads <= max_threads; nthreads *= 2)
            {
                int i = log2(nthreads);
                int k = log2(MAX_BUFFERS) - 1;
                float speedup_current = speedup[i][j][k][l];
                printf(",%.5f", speedup_current);
            }
            cout << '\n';
        }
        cout << '\n';
    }

}
