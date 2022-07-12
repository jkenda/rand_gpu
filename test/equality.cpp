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

#include <iostream>
#include <vector>
#include "../include/RNG.hpp"

using namespace std;
using std::vector;

int main(int argc, char **argv)
{
    size_t n_buffers = 8;
    size_t multi = 16;
    if (argc >= 2)
        sscanf(argv[1], "%lu", &n_buffers);
    if (argc == 3)
        sscanf(argv[2], "%lu", &multi);

    rand_gpu::RNG rng(RAND_GPU_ALGORITHM_TYCHE, n_buffers, multi);

    size_t bufsiz = rng.buffer_size();

    vector<vector<unsigned long>> buffers1(n_buffers);
    vector<vector<unsigned long>> buffers2(n_buffers);

    cout << "filling buffers 1\n";
    for (size_t i = 0; i < n_buffers; i++)
    {
        for (size_t j = 0; j < bufsiz; j++)
        {
            buffers1[i].emplace_back(rng.get_random<unsigned long>());
        }
    }

    cout << "filling buffers 2\n";
    for (size_t i = 0; i < n_buffers; i++)
    {
        for (size_t j = 0; j < bufsiz; j++)
        {
            buffers2[i].emplace_back(rng.get_random<unsigned long>());
        }
    }

    /* test how similar they are */

    // similarity within
    cout << "testing internal similarity\n";
    for (size_t i = 0; i < n_buffers; i++)
    {
        for (size_t j = i+1; j < n_buffers; j++)
        {
            long sim = 0;
            for (size_t k = 0; k < bufsiz; k++)
            {
                if (buffers1[i][k] == buffers1[j][k]) sim++;
                if (buffers2[i][k] == buffers2[j][k]) sim++;
            }
            cout << "similarity buf_" << i << " <=> buf_" << j << ": " << sim << " / " << bufsiz << '\n'; 
        }
    }

    // outside similarity
    long sim = 0;
    for (size_t i = 0; i < n_buffers; i++)
    {
        for (size_t j = 0; j < bufsiz; j++)
        {
            if (buffers1[i][j] == buffers2[i][j]) sim++;
        }
    }
    cout << "outside similarity: " << sim << " / " << n_buffers * bufsiz << '\n';
}
