
# rand_gpu

Rand_gpu is a library for efficient pseudo-random number generation on the GPU.
It was developed as part of my diploma thesis and contains 15 different RNG algorithms.

## Getting started

###  Prerequisites:
- GNU Make
- C++ compiler (GCC on Linux and Unix or Clang on MacOS)

###  Building
To get started, clone the repository and run `make fastest_multiplier`. This will build the library and a tool for finding the best parameters for the RNG.

###  Finding optimal parameters
Next, run `bin/static/fastest_multiplier` to run the tool.
This will run for quite some time. When it is finished, it will display the best parameters for each algorithm on your specific system.
Example:
```
algorithm,n_buffers,multi,speedup,buffer misses
MWC64X,2,32,5.287696,0
XORSHIFT6432*,8,64,5.259763,0
PCG6432,4,64,5.244715,0
PHILOX2X32_10,2,64,5.236758,0
LCG12864,4,8,5.067320,0
KISS09,2,8,4.982951,0
TYCHE,2,32,4.954967,0
TYCHE_I,8,32,4.942071,0
MSWS,4,32,4.922233,0
TINYMT64,2,16,4.373153,0
WELL512,2,8,3.301927,0
MRG63K3A,8,64,3.012500,0
LFIB,16,4,2.337198,87
RAN2,16,8,2.147490,84
MT19937,2,8,1.262516,88
```
As we can see, the fastest algorithm here is `MWC64X`, however, we'll choose `TYCHE`.
Its nominal parameters are `n_buffers = 2` and `multi = 32`.

###  Usage
This is an excerpt from the example `pi.cpp`. It shows how to initialize the RNG and retrieve random numbers from it.
```C++
const int n_buffers = 2, multi = 32;

rand_gpu::RNG<RAND_GPU_ALGORITHM_TYCHE> rng0(n_buffers, multi);
rand_gpu::RNG<RAND_GPU_ALGORITHM_TINYMT64> rng1(n_buffers, multi);
rng0.discard(rng0.buffer_size() / 2);

long cnt = 0;

for (uint64_t i = 0; i < SAMPLES; i++) {
    float a = rng0.get_random<float>();
    float b = rng1.get_random<float>();
    if (a*a + b*b < 1.0f) {
        cnt++;
    }
}
double pi = (double) cnt / SAMPLES * 4;
```

The C code is just a little different:
```C
const int n_buffers = 2, multi = 32;

rand_gpu_rng rng0 = rand_gpu_new_rng(RAND_GPU_ALGORITHM_TYCHE, n_buffers, multi);
rand_gpu_rng rng1 = rand_gpu_new_rng(RAND_GPU_ALGORITHM_TINYMT64, n_buffers, multi);
rand_gpu_rng_discard(rng0, rand_gpu_rng_buffer_size(rng0) / 2);

long cnt = 0;

for (uint64_t i = 0; i < SAMPLES; i++) {
    float a = rand_gpu_rng_float(rng0);
    float b = rand_gpu_rng_float(rng1);
    if (a*a + b*b < 1.0f) {
        cnt++;
    }
}
double pi = (double) cnt / SAMPLES * 4;
```

## Examples

Examples are located in the `examples` directory and include simple algorithms that use pseudo-random numbers.
