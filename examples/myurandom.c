#include <stdio.h>
#include "../include/rand_gpu.h"

int main()
{
    rand_gpu_rng rng = rand_gpu_new_rng(RAND_GPU_ALGORITHM_TINYMT64, 4, 4);

    while (1)
    {
        putchar(rand_gpu_rng_8b(rng));
    }

    rand_gpu_delete_all();
}
