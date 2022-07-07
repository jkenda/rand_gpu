#include <iostream>
#include <cstdint>
#include "../src/RNG.hpp"

using namespace std;

template <typename T>
void print_numbers(rand_gpu::RNG &rng, const size_t num, const char *name)
{
    cout << name << ": " << 8 * sizeof(T) << " bits\n";
    for (int i = 0; i < 500; i++) {
        cout << to_string(rng.get_random<T>()) << ' ';
    }
    cout << "\n\n";
}

int main()
{
    rand_gpu::RNG rng(4, 1);

    print_numbers<uint8_t>(rng, 500, "u8");
    print_numbers<int8_t>(rng, 500, "i8");
    print_numbers<uint16_t>(rng, 500, "u16");
    print_numbers<int16_t>(rng, 500, "i16");
    print_numbers<uint32_t>(rng, 500, "u32");
    print_numbers<int32_t>(rng, 500, "i32");
    print_numbers<uint64_t>(rng, 500, "u64");
    print_numbers<int64_t>(rng, 500, "i64");

    cout << "memory usage: " << rand_gpu::memory_usage() << " bytes\n";
}
