#include "RNG.hpp"
#include "RNG_private.hpp"

using namespace std;

namespace rand_gpu
{
    RNG::RNG(size_t multi)
    :   d_ptr_(make_unique<RNG_private>(multi))
    {
    }

    RNG::~RNG() = default;

    template <typename T>
    T RNG::get_random()
    {
        return d_ptr_->get_random<T>();
    }

    /*
    instantiate templates for all primitives
    */

    template unsigned long long RNG::get_random<unsigned long long>();
    template unsigned long      RNG::get_random<unsigned long>();
    template unsigned int       RNG::get_random<unsigned int>();
    template unsigned short     RNG::get_random<unsigned short>();
    template unsigned char      RNG::get_random<unsigned char>();
    template long long RNG::get_random<long long>();
    template long      RNG::get_random<long>();
    template int       RNG::get_random<int>();
    template short     RNG::get_random<short>();
    template char      RNG::get_random<char>();

    size_t RNG::buffer_size()
    {
        return d_ptr_->buffer_size();
    }

} // namespace rand_gpu
