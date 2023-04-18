/**
 * @file equality.c
 * @author Jakob Kenda (kenda.jakob@gmail.com)
 * @brief Tests whether the random number buffers change over time
 *        The result should be 0.0 for a <=> b, a <=> c and b <=> c
 * @version 0.3
 * @date 2022-04-09
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include "../include/RNG.hpp"

using namespace std;
using std::vector;

template <int A>
void equal_with_same_seed(size_t n_buffers, size_t multi)
{
    equal_with_same_seed<A-1>(n_buffers, multi);
    constexpr rand_gpu_algorithm algorithm = (rand_gpu_algorithm) A;

    // two RNGs with the same seed should yield the same sequence of numbers
    cout << "equality between 2 RNGs: " << rand_gpu_algorithm_name(algorithm, false) << "... "; flush(cout);

    // create seed
    random_device dev;
    uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
    uint64_t seed = dist(dev);

    // initialize RNG
    rand_gpu::RNG<algorithm> rng0(n_buffers, multi, seed);
    rand_gpu::RNG<algorithm> rng1(n_buffers, multi, seed);
    size_t bufsiz = rng0.buffer_size();


    size_t inequality = 0;
    for (size_t i = 0; i < n_buffers * multi * bufsiz * 23; i++)
    {
        if (rng0.template get_random<uint8_t>() != rng1.template get_random<uint8_t>())
            inequality++;
    }

    if (inequality == 0) cout << "OK: complete equality\n";
    else cout << "ERR: inequality = " << inequality << " / " << n_buffers * bufsiz << '\n';
}

template <>
void equal_with_same_seed<-1>(size_t n_buffers, size_t multi)
{
}

int main(int argc, char **argv)
{
    size_t n_buffers = 2;
    size_t multi = 1;
    if (argc >= 2)
        sscanf(argv[1], "%lu", &n_buffers);
    if (argc == 3)
        sscanf(argv[2], "%lu", &multi);

    rand_gpu::RNG<RAND_GPU_ALGORITHM_MT19937> rng(n_buffers, multi);

    size_t bufsiz = rng.buffer_size();

    vector<vector<unsigned long>> buffers1(n_buffers);
    vector<vector<unsigned long>> buffers2(n_buffers);

    cout << "buffer similarity test:\n";

    for (size_t i = 0; i < n_buffers; i++)
    {
        for (size_t j = 0; j < bufsiz; j++)
        {
            buffers1[i].emplace_back(rng.get_random<unsigned long>());
        }
    }

    for (size_t i = 0; i < n_buffers; i++)
    {
        for (size_t j = 0; j < bufsiz; j++)
        {
            buffers2[i].emplace_back(rng.get_random<unsigned long>());
        }
    }

    /* test how similar they are */

    // similarity within
    cout << "\tinternal similarity... ";
    size_t similarity = 0;
    for (size_t i = 0; i < n_buffers; i++)
    {
        for (size_t j = i+1; j < n_buffers; j++)
        {
            size_t sim = 0;
            for (size_t k = 0; k < bufsiz; k++)
            {
                if (buffers1[i][k] == buffers1[j][k]) sim++;
                if (buffers2[i][k] == buffers2[j][k]) sim++;
            }
            if (sim != 0) similarity++;
        }
    }
    if (similarity == 0) cout << "OK: similarity = 0\n";
    else cout << "ERR: similarity = " << similarity << " / " << n_buffers*n_buffers;

    cout << "\toutside similarity ... ";
    similarity = 0;
    for (size_t i = 0; i < n_buffers; i++)
    {
        for (size_t j = 0; j < bufsiz; j++)
        {
            if (buffers1[i][j] == buffers2[i][j]) similarity++;
        }
    }
    if (similarity == 0) cout << "OK: similarity = 0\n";
    else cout << "ERR: similarity =" << similarity << " / " << n_buffers * bufsiz << '\n';

    equal_with_same_seed<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>(n_buffers, multi);
}
