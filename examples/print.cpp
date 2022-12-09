#include <iostream>
#include <cstdint>
#include "../include/RNG.hpp"

using namespace std;

template <typename T>
void print_numbers(rand_gpu::RNG<RAND_GPU_ALGORITHM_MT19937> &rng, const size_t num, const char *name)
{
    cout << name << ": " << 8 * sizeof(T) << " bits\n";
    for (int i = 0; i < 256; i++) {
        cout << to_string(rng.get_random<T>()) << ' ';
    }
    cout << "\n\n";
}

int main()
{
    rand_gpu::RNG<RAND_GPU_ALGORITHM_MT19937> rng(2, 1, 0);

    print_numbers<uint8_t>(rng, 256, "u8");
    print_numbers<int8_t>(rng, 256, "i8");
    print_numbers<uint16_t>(rng, 256, "u16");
    print_numbers<int16_t>(rng, 256, "i16");
    print_numbers<uint32_t>(rng, 256, "u32");
    print_numbers<int32_t>(rng, 256, "i32");
    print_numbers<uint64_t>(rng, 256, "u64");
    print_numbers<int64_t>(rng, 256, "i64");

    cout << "memory usage: " << rand_gpu::memory_usage() << " bytes\n";
}
