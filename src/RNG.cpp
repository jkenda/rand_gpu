/**
 * @file RNG.cpp
 * @author Jakob Kenda (kenda.jakob@gmail.com)
 * @brief Source file for RNG/rand_gpu C/C++ library
 * @version 0.3
 * @date 2022-04-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "../include/RNG.hpp"
#include "../include/rand_gpu.h"

#include <cstdint>
#include <vector>
#include <unordered_set>
#include <mutex>
#include <condition_variable>
#include <random>
#include <iostream>
#include <cstring>
#include <atomic>
#include <chrono>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#include "include/cl.hpp"
#include "../kernel.hpp"

using namespace std;
using chrono::nanoseconds;
using chrono::system_clock;

struct Buffer
{
    uint8_t *data = nullptr;
    bool ready = true;
    mutex ready_lock;
    cl::Event ready_event;
    condition_variable ready_cond;
    system_clock::time_point start_time;
    RNG_private *rng;

    ~Buffer() { delete[] data; }
    inline void alloc(const size_t size) { data = new uint8_t[size]; }
};

atomic<nanoseconds> &operator+=(atomic<nanoseconds> &a, nanoseconds b)
{
    a.exchange(a.load() + b);
    return a;
}


static const size_t STATE_SIZES[] = {
    4 * sizeof(cl_ulong),                                        // KISS09,
    2 * sizeof(cl_ulong),                                        // LCG12864,
    17 * sizeof(cl_ulong) + 2 * sizeof(cl_char),                 // LFIB,
    6 * sizeof(cl_long),                                         // MRG63K3A,
    max(sizeof(cl_ulong) + 2 * sizeof(cl_uint2), sizeof(ulong)), // MSWS,
    624 * sizeof(cl_uint) + sizeof(cl_int),                      // MT19937,
    max(sizeof(cl_ulong), 2 * sizeof(cl_uint)),                  // MWC64X,
    sizeof(cl_ulong),                                            // PCG6432,
    max(sizeof(cl_ulong), sizeof(cl_uint)),                      // PHILOX2X32_10,
    35 * sizeof(cl_int),                                         // RAN2,
    3 * sizeof(cl_ulong) + 2 * sizeof(cl_uint),                  // TINYMT64,
    max(4 * sizeof(cl_uint), sizeof(cl_ulong)),                  // TYCHE,
    max(4 * sizeof(cl_uint), sizeof(cl_ulong)),                  // TYCHE_I,
    17 * sizeof(cl_uint),                                        // WELL512,
    sizeof(cl_ulong),                                            // XORSHIFT6432STAR,
};

static const char *INIT_KERNEL_NAMES[] = {
    "kiss09_init",           // KISS09,
    "lcg12864_init",         // LCG12864,
    "lfib_init",             // LFIB,
    "mrg63k3a_init",         // MRG63K3A,
    "msws_init",             // MSWS,
    "mt19937_init",          // MT19937,
    "mwc64x_init",           // MWC64X,
    "pcg6432_init",          // PCG6432,
    "philox2x32_10_init",    // PHILOX2X32_10,
    "ran2_init",             // RAN2,
    "tinymt64_init",         // TINYMT64,
    "tyche_init",            // TYCHE,
    "tyche_i_init",          // TYCHE_I,
    "well512_init",          // WELL512,
    "xorshift6432star_init", // XORSHIFT6432STAR,
};

static const char *GENERATE_KERNEL_NAMES[] = {
    "kiss09_generate",           // KISS09,
    "lcg12864_generate",         // LCG12864,
    "lfib_generate",             // LFIB,
    "mrg63k3a_generate",         // MRG63K3A,
    "msws_generate",             // MSWS,
    "mt19937_generate",          // MT19937,
    "mwc64x_generate",           // MWC64X,
    "pcg6432_generate",          // PCG6432,
    "philox2x32_10_generate",    // PHILOX2X32_10,
    "ran2_generate",             // RAN2,
    "tinymt64_generate",         // TINYMT64,
    "tyche_generate",            // TYCHE,
    "tyche_i_generate",          // TYCHE_I,
    "well512_generate",          // WELL512,
    "xorshift6432star_generate", // XORSHIFT6432STAR,
};

static const float FLOAT_MULTIPLIER = 1.0f / static_cast<float>(UINT32_MAX);
static const double DOUBLE_MULTIPLIER = 1.0 / static_cast<double>(UINT64_MAX);
static const long double LONG_DOUBLE_MULTIPLIER = 1.0l / static_cast<long double>(UINT64_MAX);
static const float MILLION = static_cast<float>(1'000'000);
static const size_t N_ALGORITHMS = RAND_GPU_ALGORITHM_XORSHIFT6432STAR - RAND_GPU_ALGORITHM_KISS09 + 1;


static mutex __constructor_lock;
static bool __initialized = false;

static vector<cl::Device> __devices;
static size_t __device_i = 0;
static atomic<size_t> __memory_usage(0);

static mt19937_64 __generator;
static cl::Context __context;
static cl::Program __programs[N_ALGORITHMS];
static bool __program_built[N_ALGORITHMS];
static nanoseconds __compilation_times[N_ALGORITHMS];

static atomic<nanoseconds> __sum_init_time_deleted(nanoseconds::zero());
static atomic<nanoseconds> __sum_avg_transfer_time_deleted(nanoseconds::zero());

static atomic<size_t> __sum_buffer_misses_deleted(0);
static atomic<size_t> __sum_buffer_switches_deleted(0);
static atomic<size_t> __n_deleted_rngs(0);

static mutex __rngs_lock;
static mutex __delete_all_lock;
static unordered_set<RNG_private *> __rngs;


struct RNG_private
{
    cl::CommandQueue _queue;
    cl::Kernel _k_generate;
    cl::Buffer _states_buffer;
    cl::Buffer _device_buffer;

    uint32_t _max_cu;
    size_t _max_wg_size;

    cl::NDRange _global_size;
    size_t _buffer_size;
    size_t _buf_limit;

    const size_t _n_buffers;
    vector<Buffer> _host_buffers;
    uint_fast8_t _active_buf = 0;

    uint_fast64_t _buf_offset = 0;

    nanoseconds _init_time = nanoseconds::zero();
    nanoseconds _gpu_transfer_time_total = nanoseconds::zero();

    uint_fast64_t _n_gpu_transfers = 0;
    uint_fast64_t _n_buffer_switches = 0;
    uint_fast64_t _n_buffer_misses = 0;


    RNG_private(size_t n_buffers = 2, size_t multi = 1, rand_gpu_algorithm algorithm = RAND_GPU_ALGORITHM_TYCHE,
        bool use_custom_seed = false, uint64_t custom_seed = 0)
    :
        _n_buffers(max(n_buffers, 2LU)),
        _host_buffers(max(n_buffers, 2LU))
    {
        cl_ulong seed;
        cl::Device device;
        cl::Program &program = __programs[algorithm];
        auto start = chrono::system_clock::now();

        {
            lock_guard<mutex> lock(__constructor_lock);

            if (!__initialized)
            {
                cl::Platform platform;

                // get platforms and devices
                try
                {
                    cl::Platform::get(&platform);
                    platform.getDevices(CL_DEVICE_TYPE_GPU, &__devices);
                }
                catch (const cl::Error &err)
                {
                    cerr << "No openCL platforms/devices found!\n";
                    throw err;
                }

                // create context
                __context = cl::Context(__devices);

                // initialize host RNG for generating seeds
                __generator = mt19937_64(system_clock::now().time_since_epoch().count());

                __initialized = true;
            }

            // generate seed
            if (use_custom_seed)
                seed = custom_seed;
            else
                seed = __generator();
            
            // distribute devices among instances
            device = __devices[__device_i];
            __device_i = (__device_i + 1) % __devices.size();

            if (!__program_built[algorithm])
            {
                // build program
                program = cl::Program(__context, KERNEL_SOURCE[algorithm]);

                try
                {
                    auto build_start = system_clock::now();
                    program.build(__devices);
                    __compilation_times[algorithm] = system_clock::now() - build_start;
                }
                catch (const cl::Error &err)
                {
                    // print buildlog if build failed
                    for (const cl::Device &device : __devices) 
                    {
                        if (program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) != CL_SUCCESS)
                        {
                            string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                            cerr << buildlog << '\n';
                            throw err;
                        }
                    }
                }
                __program_built[algorithm] = true;
            }

        }

        // create command queue
        _queue = cl::CommandQueue(__context, device);

        // get device info
        _max_cu = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        _max_wg_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

        _global_size = cl::NDRange(max(multi, 1LU) * _max_cu * _max_wg_size);
        _buffer_size = _global_size[0] * sizeof(cl_ulong);
        _buf_limit = _buffer_size - sizeof(long double);

        // create host and device buffers
        _states_buffer = cl::Buffer(__context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, _global_size[0] * STATE_SIZES[algorithm]);
        _device_buffer = cl::Buffer(__context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, _buffer_size);

        for (Buffer &buf : _host_buffers)
        {
            buf.rng = this;
            buf.alloc(_buffer_size);
        }

        // initialize RNG
        cl::Kernel k_init(program, INIT_KERNEL_NAMES[algorithm]);
        cl::Event initialized;
        k_init.setArg(0, _states_buffer);

        switch (algorithm)
        {
        case RAND_GPU_ALGORITHM_TYCHE:
        case RAND_GPU_ALGORITHM_TYCHE_I:
            // a single seed for all threads
            k_init.setArg(1, sizeof(cl_ulong), &seed);
            _queue.enqueueNDRangeKernel(k_init, 0, _global_size);
            break;
        case RAND_GPU_ALGORITHM_MT19937:
            // 32-bit seeds
            {
                mt19937 seed_generator(seed);
                seed_generator.discard(_global_size[0]);
                vector<uint32_t> seeds(_global_size[0]);
                for (size_t i = 0; i < _global_size[0]; i++)
                    seeds[i] = seed_generator();
                cl::Buffer cl_seeds(_queue, seeds.rbegin(), seeds.rend(), true, false);
                k_init.setArg(1, cl_seeds);
                cl::Event e;
                _queue.enqueueNDRangeKernel(k_init, 0, _global_size);
            }
            break;
        default:
            // 64-bit seeds
            {
                mt19937_64 seed_generator(seed);
                vector<uint64_t> seeds(_global_size[0]);
                for (size_t i = 0; i < _global_size[0]; i++)
                    seeds[i] = seed_generator();
                cl::Buffer cl_seeds(_queue, seeds.begin(), seeds.end(), true, true);
                k_init.setArg(1, cl_seeds);
                _queue.enqueueNDRangeKernel(k_init, 0, _global_size);
            }
        }

        // create kernel
        _k_generate = cl::Kernel(program, GENERATE_KERNEL_NAMES[algorithm]);
        _k_generate.setArg(0, _states_buffer);
        _k_generate.setArg(1, _device_buffer);

        // fill all buffers
        for (size_t i = 0; i < _n_buffers; i++)
        {
            _queue.enqueueNDRangeKernel(_k_generate, 0, _global_size);
            _queue.enqueueReadBuffer(_device_buffer, true, 0, _buffer_size, _host_buffers[i].data);
        }

        // generate future numbers
        _queue.enqueueNDRangeKernel(_k_generate, 0, _global_size);

        _queue.flush();
        _queue.finish();

        _init_time = chrono::system_clock::now() - start;
        __memory_usage += _n_buffers * _buffer_size;

        lock_guard<mutex> rngs_lock(__rngs_lock);
        __rngs.insert(this);
    }

    ~RNG_private()
    {
        // ensure set_flag won't try to access a deleted Buffer
        for (Buffer &buffer : _host_buffers)
        {
            unique_lock<mutex> lock(buffer.ready_lock);
            buffer.ready_cond.wait(lock, [&] { return buffer.ready; });
        }

        _queue.flush();
        _queue.finish();

        // set global variables
        __memory_usage -= _n_buffers * _buffer_size;
        __sum_avg_transfer_time_deleted += avg_gpu_transfer_time();
        __sum_init_time_deleted += _init_time;
        __sum_buffer_switches_deleted += _n_buffer_switches;
        __sum_buffer_misses_deleted += _n_buffer_misses;
        __n_deleted_rngs++;

        lock_guard<mutex> rngs_lock(__rngs_lock);
        __rngs.erase(this);
    }


    template <typename T>
    T get_random()
    {
        Buffer &active_host_buf = _host_buffers[_active_buf];

        // just switched buffers - wait for buffer to be ready
        if (_buf_offset == 0)
        {
            unique_lock<mutex> lock(active_host_buf.ready_lock);
            if (!active_host_buf.ready) _n_buffer_misses++;
            active_host_buf.ready_cond.wait(lock, [&] { return active_host_buf.ready; });
        }

        // retrieve number from buffer
        T num;
        memcpy(&num, &active_host_buf.data[_buf_offset], sizeof(T));
        _buf_offset += sizeof(T);

        // out of numbers in current buffer
        if (_buf_offset >= _buf_limit)
        {
            active_host_buf.ready = false;

            // enqueue reading data, generating future numbers
            _queue.enqueueReadBuffer(_device_buffer, false, 0, _buffer_size, active_host_buf.data, nullptr, &active_host_buf.ready_event);
            _queue.enqueueNDRangeKernel(_k_generate, 0, _global_size);
            active_host_buf.start_time = system_clock::now();
            active_host_buf.ready_event.setCallback(CL_COMPLETE, set_flag, &active_host_buf);

            // switch active buffer
            _active_buf = (_active_buf + 1) % _n_buffers;
            _buf_offset = 0;
            _n_buffer_switches++;
        }

        return num;
    }

    static void set_flag(cl_event _e, cl_int _s, void *ptr)
    {
        // notify that buffer is ready
        Buffer *buffer = static_cast<Buffer *>(ptr);
        lock_guard<mutex> lock(buffer->ready_lock);
        buffer->rng->_gpu_transfer_time_total += system_clock::now() - buffer->start_time;
        buffer->rng->_n_gpu_transfers++;
        buffer->ready = true;
        buffer->ready_cond.notify_one();
    }

    nanoseconds avg_gpu_transfer_time() const
    {
        if (_n_gpu_transfers > 0)
            return _gpu_transfer_time_total / _n_gpu_transfers;
        else
            return chrono::nanoseconds::zero();
    }

};


/*
template specialization
*/

template <>
float RNG_private::get_random<float>()
{
    return FLOAT_MULTIPLIER * get_random<unsigned int>();
}

template <>
double RNG_private::get_random<double>()
{
    return DOUBLE_MULTIPLIER * get_random<unsigned long>();
}

template <>
long double RNG_private::get_random<long double>()
{
    return LONG_DOUBLE_MULTIPLIER * get_random<unsigned long long>();
}

template <>
bool RNG_private::get_random<bool>()
{
    return get_random<uint32_t>() > (UINT32_MAX / 2U) ? true : false;
}


/*
C wrapper functions
*/

extern "C" {

rand_gpu_rng *rand_gpu_new_default()
{
    return new RNG_private;
}

rand_gpu_rng *rand_gpu_new(enum rand_gpu_algorithm algorithm, size_t n_buffers, size_t buffer_multi)
{
    return new RNG_private(n_buffers, buffer_multi, algorithm);
}

rand_gpu_rng *rand_gpu_new_with_seed(uint64_t seed, enum rand_gpu_algorithm algorithm, size_t n_buffers, size_t buffer_multi)
{
    return new RNG_private(n_buffers, buffer_multi, algorithm, true, seed);
}

void rand_gpu_delete(rand_gpu_rng *rng)
{
    delete (RNG_private *) rng;
}

void rand_gpu_delete_all()
{
    lock_guard<mutex> lock(__delete_all_lock);
    for (auto it = __rngs.begin(); it != __rngs.end();)
    {
        delete *(it++);
    }
}

size_t rand_gpu_rng_buffer_size(const rand_gpu_rng *rng)           { return ((const RNG_private *) rng)->_buffer_size; }
size_t rand_gpu_rng_buffer_switches(const rand_gpu_rng *rng)       { return ((const RNG_private *) rng)->_n_buffer_switches; }
size_t rand_gpu_rng_buffer_misses(const rand_gpu_rng *rng)         { return ((const RNG_private *) rng)->_n_buffer_misses; }
float  rand_gpu_rng_init_time(const rand_gpu_rng *rng)             { return ((const RNG_private *) rng)->_init_time.count() / MILLION; }
float  rand_gpu_rng_avg_gpu_transfer_time(const rand_gpu_rng *rng) { return ((const RNG_private *) rng)->avg_gpu_transfer_time().count() / MILLION; }

size_t rand_gpu_memory_usage()          { return __memory_usage; }
size_t rand_gpu_buffer_switches()       { return rand_gpu::buffer_switches(); }
size_t rand_gpu_buffer_misses()         { return rand_gpu::buffer_misses(); }
float  rand_gpu_avg_init_time()         { return rand_gpu::avg_init_time().count() / MILLION; }
float  rand_gpu_avg_gpu_transfer_time() { return rand_gpu::avg_gpu_transfer_time().count() / MILLION; }

float rand_gpu_compilation_time(rand_gpu_algorithm algorithm) { return __compilation_times[algorithm].count() / MILLION; }
const char *rand_gpu_algorithm_name(rand_gpu_algorithm algorithm, bool description) { return rand_gpu::algorithm_name(algorithm, description); }

uint64_t rand_gpu_u64(rand_gpu_rng *rng) { return ((RNG_private *) rng)->get_random<uint64_t>(); }
uint32_t rand_gpu_u32(rand_gpu_rng *rng) { return ((RNG_private *) rng)->get_random<uint32_t>(); }
uint16_t rand_gpu_u16(rand_gpu_rng *rng) { return ((RNG_private *) rng)->get_random<uint16_t>(); }
uint8_t  rand_gpu_u8(rand_gpu_rng *rng)  { return ((RNG_private *) rng)->get_random<uint8_t>();  }

float       rand_gpu_float(rand_gpu_rng *rng)       { return ((RNG_private *) rng)->get_random<float>();       }
double      rand_gpu_double(rand_gpu_rng *rng)      { return ((RNG_private *) rng)->get_random<double>();      }
long double rand_gpu_long_double(rand_gpu_rng *rng) { return ((RNG_private *) rng)->get_random<long double>(); }

}


/*
RNG definitions
*/

namespace rand_gpu
{
    template <rand_gpu_algorithm A>
    RNG<A>::RNG()
    :
        d_ptr_(new RNG_private(2, 1, A, false))
    {
    }

    template <rand_gpu_algorithm A>
    RNG<A>::RNG(size_t n_buffers, size_t multi)
    :
        d_ptr_(new RNG_private(n_buffers, multi, A, false))
    {
    }

    template <rand_gpu_algorithm A>
    RNG<A>::RNG(unsigned long seed, size_t n_buffers, size_t multi)
    :
        d_ptr_(new RNG_private(n_buffers, multi, A, true, seed))
    {
    }

    template <rand_gpu_algorithm A>
    RNG<A>::~RNG()
    {
    }

    template <rand_gpu_algorithm A>
    RNG<A>::RNG(RNG&&) = default;
    template <rand_gpu_algorithm A>
    RNG<A> &RNG<A>::operator=(RNG&&) = default;

    template <rand_gpu_algorithm A>
    template <typename T>
    T RNG<A>::get_random()
    {
        return d_ptr_->get_random<T>();
    }

    template <rand_gpu_algorithm A>
    size_t RNG<A>::buffer_size() const
    {
        return d_ptr_->_buffer_size;
    }

    template <rand_gpu_algorithm A>
    size_t RNG<A>::buffer_switches() const
    {
        return d_ptr_->_n_buffer_switches;
    }

    template <rand_gpu_algorithm A>
    size_t RNG<A>::buffer_misses() const
    {
        return d_ptr_->_n_buffer_misses;
    }

    template <rand_gpu_algorithm A>
    nanoseconds RNG<A>::init_time() const
    {
        return d_ptr_->_init_time;
    }

    template <rand_gpu_algorithm A>
    std::chrono::nanoseconds RNG<A>::avg_gpu_transfer_time() const
    {
        return d_ptr_->avg_gpu_transfer_time();
    }

    size_t memory_usage()
    {
        return __memory_usage;
    }


    nanoseconds avg_gpu_transfer_time()
    {
        nanoseconds sum(0);
        lock_guard<mutex> lock(__rngs_lock);

        // add the averages of active RNGs
        for (RNG_private *rng : __rngs)
        {
            sum += rng->avg_gpu_transfer_time();
        }

        // add the averages of deleted RNGs
        sum += __sum_avg_transfer_time_deleted.load();
        return sum / (__rngs.size() + __n_deleted_rngs);
    }

    nanoseconds avg_init_time()
    {
        nanoseconds sum(0);
        lock_guard<mutex> lock(__rngs_lock);

        // add the averages of active RNGs
        for (RNG_private *rng : __rngs)
        {
            sum += rng->_init_time;
        }

        // add the averages of deleted RNGs
        sum += __sum_init_time_deleted.load();
        return sum / (__rngs.size() + __n_deleted_rngs);
    }

    size_t buffer_switches()
    {
        size_t sum = 0;
        lock_guard<mutex> lock(__rngs_lock);

        // add the averages of active RNGs
        for (RNG_private *rng : __rngs)
        {
            sum += rng->_n_buffer_switches;
        }

        // add the averages of deleted RNGs
        sum += __sum_buffer_switches_deleted;
        return sum;
    }

    size_t buffer_misses()
    {
        size_t sum = 0;
        lock_guard<mutex> lock(__rngs_lock);

        // add the averages of active RNGs
        for (RNG_private *rng : __rngs)
        {
            sum += rng->_n_buffer_misses;
        }

        // add the averages of deleted RNGs
        sum += __sum_buffer_misses_deleted;
        return sum;
    }

    std::chrono::nanoseconds compilation_time(rand_gpu_algorithm algorithm)
    {
        return __compilation_times[algorithm];
    }

    const char *algorithm_name(rand_gpu_algorithm algorithm, bool description)
    {
        const char *names[] = {
            "KISS09",
            "LCG12864",
            "LFIB",
            "MRG63K3A",
            "MSWS",
            "MT19937",
            "MWC64X",
            "PCG6432",
            "PHILOX2X32_10",
            "RAN2",
            "TINYMT64",
            "TYCHE",
            "TYCHE_I",
            "WELL512",
            "XORSHIFT6432STAR",
        };

        const char *descriptions[] = {
            "Kiss09 - KISS (Keep It Simple, Stupid)",
            "LCG12864 - 128-bit Linear Congruential Generator",
            "LFib - Multiplicative Lagged Fibbonaci generator (lowest bit is always 1)",
            "MRG63K3A (Multiple Recursive Generator)",
            "MSWS - Middle Square Weyl Sequence",
            "MT19937 - Mersenne twister: a 623-dimensionally equidistributed uniform pseudo-random number generator",
            "MWC64x - 64-bit Multiply With Carry generator that returns 32-bit numbers that are xor of lower and upper 32-bit numbers",
            "PCG6432 - 64-bit Permutated Congruential generator (PCG-XSH-RR)",
            "Philox2x32_10",
            "Ran2 - a L'Ecuyer combined recursive generator with a 32-element shuffle-box (from Numerical Recipes)",
            "Tinymt64 - Tiny Mersenne twister",
            "Tyche - 512-bit Tyche (Well-Equidistributed Long-period Linear RNG)",
            "Tyche-i - a faster variant of Tyche with a shorter period",
            "WELL512 - 512-bit WELL (Well-Equidistributed Long-period Linear) RNG",
            "xorshift6432* - 64-bit xorshift* generator that returns 32-bit values",
        };
        return description ? descriptions[algorithm] : names[algorithm];
    }

    /*
    instantiate templates for all algorithms and primitives
    */

    template class RNG<RAND_GPU_ALGORITHM_KISS09>;
    template class RNG<RAND_GPU_ALGORITHM_LCG12864>;
    template class RNG<RAND_GPU_ALGORITHM_LFIB>;
    template class RNG<RAND_GPU_ALGORITHM_MRG63K3A>;
    template class RNG<RAND_GPU_ALGORITHM_MSWS>;
    template class RNG<RAND_GPU_ALGORITHM_MT19937>;
    template class RNG<RAND_GPU_ALGORITHM_MWC64X>;
    template class RNG<RAND_GPU_ALGORITHM_PCG6432>;
    template class RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>;
    template class RNG<RAND_GPU_ALGORITHM_RAN2>;
    template class RNG<RAND_GPU_ALGORITHM_TINYMT64>;
    template class RNG<RAND_GPU_ALGORITHM_TYCHE>;
    template class RNG<RAND_GPU_ALGORITHM_TYCHE_I>;
    template class RNG<RAND_GPU_ALGORITHM_WELL512>;
    template class RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>;

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<long double>();

}
