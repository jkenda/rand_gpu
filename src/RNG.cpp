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
#include <array>
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
using chrono::duration_cast, chrono::nanoseconds, chrono::system_clock;

struct Buffer
{
    uint8_t *data = nullptr;
    bool ready = true;
    mutex ready_lock;
    cl::Event ready_event;
    condition_variable ready_cond;

    inline void alloc(const size_t size) { data = new uint8_t[size]; }
    ~Buffer() { delete[] data; }
};


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


static mutex __constructor_lock;
static mutex __init_lock;
static mutex __queue_lock;

static vector<cl::Device> __devices;
static size_t __device_i = 0;
static atomic<size_t> __mem_all = 0;
static bool __initialized = false;

static mt19937_64 __generator;
static cl::Context __context;
static cl::Device  __device;
static cl::Program __program;

static cl::NDRange __global_size;
static size_t __buffer_size;
static size_t __buf_limit;


struct RNG_private
{
    cl::CommandQueue _queue;
    cl::Kernel _k_generate;
    cl::Buffer _state_buf;
    cl::Buffer _device_buffer;

    const size_t _n_buffers;
    vector<Buffer> _host_buffers;
    uint_fast8_t _active_buf = 0;

    size_t _buf_offset = sizeof(long double);

    size_t _buffer_misses = 0;
    float _init_time;

    RNG_private(size_t n_buffers = 2, size_t multi = 1, rand_gpu_algorithm algorithm = RAND_GPU_ALGORITHM_TYCHE,
        bool use_custom_seed = false, const unsigned long custom_seed = 0)
    :
        _n_buffers(max(n_buffers, 2LU)),
        _host_buffers(max(n_buffers, 2LU))
    {
        cl_ulong seed;
        auto start = chrono::high_resolution_clock::now();

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
                    __device = __devices[__device_i];
                }
                catch (const cl::Error& err)
                {
                    cerr << "No openCL platforms/devices found!\n";
                    throw err;
                }

                // create context
                __context = cl::Context(__devices);

                // build program
                cl::Program::Sources sources(1, make_pair(KERNEL_SOURCE, strlen(KERNEL_SOURCE)));
                __program = cl::Program(__context, sources);

                try
                {
                    __program.build(__devices);
                }
                catch (const cl::Error& err)
                {
                    // print buildlog if build failed
                    for (const cl::Device& dev : __devices) 
                    {
                        if (__program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev) != CL_SUCCESS)
                        {
                            string buildlog = __program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                            cerr << buildlog << '\n';
                            throw err;
                        }
                    }
                }

                // initialize host RNG for generating seeds
                if (!use_custom_seed)
                    __generator = mt19937_64(system_clock::now().time_since_epoch().count());

                // get device info
                uint32_t max_cu = __device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
                size_t max_wg_size = __device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

                __global_size = cl::NDRange(max(multi, 1LU) * max_cu * max_wg_size);
                __buffer_size = __global_size[0] * sizeof(cl_ulong);
                __buf_limit = __buffer_size - sizeof(long double);

                __initialized = true;
            }

            // generate seed
            if (use_custom_seed)
                seed = custom_seed;
            else
                seed = __generator();
            
            // distribute devices among instances
            __device = __devices[__device_i];
            __device_i = (__device_i + 1) % __devices.size();

            // create command queue
            _queue = cl::CommandQueue(__context, __device);
        }

        // increase total memory usage counter
        __mem_all += _n_buffers * __buffer_size;

        // resize host buffers, create device buffers
        _state_buf = cl::Buffer(__context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, __global_size[0] * STATE_SIZES[algorithm]);
        _device_buffer = cl::Buffer(__context, CL_MEM_WRITE_ONLY, __buffer_size);
        vector<cl::Buffer> temp_buffers(_n_buffers-1, cl::Buffer(__context, CL_MEM_WRITE_ONLY, __buffer_size));
        for (Buffer &buf : _host_buffers)
        {
            buf.alloc(__buffer_size);
        }

        // initialize RNG
        cl::Kernel k_init(__program, INIT_KERNEL_NAMES[algorithm]);
        cl::Event initialized;
        k_init.setArg(0, _state_buf);

        switch (algorithm)
        {
        case RAND_GPU_ALGORITHM_TYCHE:
        case RAND_GPU_ALGORITHM_TYCHE_I:
            k_init.setArg(1, sizeof(cl_ulong), &seed);
            _queue.enqueueNDRangeKernel(k_init, 0, __global_size);
            break;
        default:
            {
                ranlux48 s(seed);
                vector<uint64_t> seeds;
                seeds.reserve(__global_size[0]);
                for (size_t i = 0; i < __global_size[0]; i++)
                {
                    seeds.emplace_back(s());
                }
                cl::Buffer cl_seeds(_queue, seeds.begin(), seeds.end(), true, false);
                k_init.setArg(1, cl_seeds);
                _queue.enqueueNDRangeKernel(k_init, cl::NullRange, __global_size);
            }
        }

        // create kernel
        _k_generate = cl::Kernel(__program, GENERATE_KERNEL_NAMES[algorithm]);
        _k_generate.setArg(0, _state_buf);

        // fill all buffers
        vector<cl::Event> buffers_read(_n_buffers);

        for (size_t i = 1; i < _n_buffers; i++)
        {
            _k_generate.setArg(1, temp_buffers[i-1]);
            _queue.enqueueNDRangeKernel(_k_generate, 0, __global_size);
            _queue.enqueueReadBuffer(temp_buffers[i-1], false, 0, __buffer_size, _host_buffers[i].data, nullptr, &buffers_read[i]);
        }
        _k_generate.setArg(1, _device_buffer);
        _queue.enqueueNDRangeKernel(_k_generate, 0, __global_size);
        _queue.enqueueReadBuffer(_device_buffer, false, 0, __buffer_size, _host_buffers[0].data, nullptr, &buffers_read[0]);
        cl::Event::waitForEvents(buffers_read);

        // generate future numbers
        _queue.enqueueNDRangeKernel(_k_generate, 0, __global_size);

        _init_time = (duration_cast<nanoseconds>(chrono::high_resolution_clock::now() - start).count() / static_cast<float>(1'000'000));
    }

    ~RNG_private()
    {
        __mem_all -= _n_buffers * __buffer_size;
    }


    template <typename T>
    T get_random()
    {
        Buffer &active_host_buf = _host_buffers[_active_buf];

        // just switched buffers - wait for buffer to be ready
        if (_buf_offset == 0)
        {
            unique_lock<mutex> lock(active_host_buf.ready_lock);
            if (!active_host_buf.ready) _buffer_misses++;
            active_host_buf.ready_cond.wait(lock, [&] { return active_host_buf.ready; });
        }

        // retrieve number from buffer
        T num;
        memcpy(&num, &active_host_buf.data[_buf_offset], sizeof(T));
        _buf_offset += sizeof(T);

        // out of numbers in current buffer
        if (_buf_offset >= __buf_limit)
        {
            active_host_buf.ready = false;

            // enqueue reading data, generating future numbers
            _queue.enqueueReadBuffer(_device_buffer, false, 0, __buffer_size, active_host_buf.data, nullptr, &active_host_buf.ready_event);
            _queue.enqueueNDRangeKernel(_k_generate, 0, __global_size);
            active_host_buf.ready_event.setCallback(CL_COMPLETE, set_flag, &active_host_buf);

            // switch active buffer
            _active_buf = (_active_buf + 1) % _n_buffers;
            _buf_offset = 0;
        }

        return num;
    }

    size_t buffer_size() const
    {
        return __buffer_size;
    }

    size_t buffer_misses() const
    {
        return _buffer_misses;
    }

    float init_time() const
    {
        return _init_time;
    }

    static void set_flag(cl_event _e, cl_int _s, void *data)
    {
        // notify that buffer is ready
        Buffer *buffer = (Buffer *) data;
        lock_guard<mutex> lock(buffer->ready_lock);
        buffer->ready = true;
        buffer->ready_cond.notify_one();
    }

};

size_t mem_usage()
{
    return __mem_all;
}


/*
template specialization
*/

template <>
float RNG_private::get_random<float>()
{
    return FLOAT_MULTIPLIER * get_random<uint32_t>();
}

template <>
double RNG_private::get_random<double>()
{
    return DOUBLE_MULTIPLIER * get_random<uint64_t>();
}

template <>
long double RNG_private::get_random<long double>()
{
    return LONG_DOUBLE_MULTIPLIER * get_random<uint64_t>();
}

template <>
bool RNG_private::get_random<bool>()
{
    return get_random<uint32_t>() > static_cast<uint32_t>(UINT32_MAX / 2U) ? true : false;
}


/*
C function wrappers
*/

extern "C" {

rand_gpu_rng *rand_gpu_new_default()
{
    return (rand_gpu_rng *) new RNG_private;
}

rand_gpu_rng *rand_gpu_new(enum rand_gpu_algorithm algorithm, size_t n_buffers, size_t buffer_multi)
{
    return (rand_gpu_rng *) new RNG_private(n_buffers, buffer_multi, algorithm);
}

rand_gpu_rng *rand_gpu_new_with_seed(uint64_t seed, enum rand_gpu_algorithm algorithm, size_t n_buffers, size_t buffer_multi)
{
    return (rand_gpu_rng *) new RNG_private(n_buffers, buffer_multi, algorithm, true, seed);
}

void rand_gpu_delete(rand_gpu_rng *rng)
{
    delete (RNG_private *) rng;
}

size_t rand_gpu_buffer_size(rand_gpu_rng *rng) { return static_cast<RNG_private *>(rng)->buffer_size(); }
size_t rand_gpu_buf_misses(rand_gpu_rng *rng)  { return static_cast<RNG_private *>(rng)->buffer_misses(); }
float  rand_gpu_init_time(rand_gpu_rng *rng)   { return static_cast<RNG_private *>(rng)->init_time(); }
size_t rand_gpu_memory() { return rand_gpu::memory_usage(); }


uint64_t rand_gpu_u64(rand_gpu_rng *rng) { return static_cast<RNG_private *>(rng)->get_random<uint64_t>(); }
uint32_t rand_gpu_u32(rand_gpu_rng *rng) { return static_cast<RNG_private *>(rng)->get_random<uint32_t>(); }
uint16_t rand_gpu_u16(rand_gpu_rng *rng) { return static_cast<RNG_private *>(rng)->get_random<uint16_t>(); }
uint8_t  rand_gpu_u8(rand_gpu_rng *rng)  { return static_cast<RNG_private *>(rng)->get_random<uint8_t>();  }

float       rand_gpu_float(rand_gpu_rng *rng)       { return static_cast<RNG_private *>(rng)->get_random<float>();       }
double      rand_gpu_double(rand_gpu_rng *rng)      { return static_cast<RNG_private *>(rng)->get_random<double>();      }
long double rand_gpu_long_double(rand_gpu_rng *rng) { return static_cast<RNG_private *>(rng)->get_random<long double>(); }

const char *rand_gpu_algorithm_name(rand_gpu_algorithm algorithm, bool long_name) { return rand_gpu::algorithm_name(algorithm, long_name); }

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
        delete d_ptr_;
    }

    template <rand_gpu_algorithm A>
    RNG<A>::RNG(RNG&&) = default;
    template <rand_gpu_algorithm A>
    RNG<A>& RNG<A>::operator=(RNG&&) = default;

    template <rand_gpu_algorithm A>
    template <typename T>
    T RNG<A>::get_random()
    {
        return d_ptr_->get_random<T>();
    }

    template <rand_gpu_algorithm A>
    size_t RNG<A>::buffer_size() const
    {
        return d_ptr_->buffer_size();
    }

    template <rand_gpu_algorithm A>
    size_t RNG<A>::buffer_misses() const
    {
        return d_ptr_->buffer_misses();
    }

    template <rand_gpu_algorithm A>
    float RNG<A>::init_time() const
    {
        return d_ptr_->init_time();
    }

    size_t memory_usage()
    {
        return mem_usage();
    }

    const char *algorithm_name(rand_gpu_algorithm algorithm, bool long_name)
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

        const char *long_names[] = {
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
            "Tinymt64 - Tiny mersenne twister",
            "Tyche - 512-bit Tyche (Well-Equidistributed Long-period Linear RNG)",
            "Tyche-i - a faster variant of Tyche with a shorter period",
            "WELL512 - 512-bit WELL (Well-Equidistributed Long-period Linear) RNG",
            "xorshift6432* - 64-bit xorshift* generator that returns 32-bit values",
        };
        return long_name ? long_names[algorithm] : names[algorithm];
    }

    /*
    instantiate templates for all algorithms and primitives
    */

    template RNG<RAND_GPU_ALGORITHM_KISS09>::RNG();
    template RNG<RAND_GPU_ALGORITHM_KISS09>::RNG(size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_KISS09>::RNG(unsigned long seed, size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_KISS09>::~RNG();
    template size_t RNG<RAND_GPU_ALGORITHM_KISS09>::buffer_size() const;
    template size_t RNG<RAND_GPU_ALGORITHM_KISS09>::buffer_misses() const;
    template float RNG<RAND_GPU_ALGORITHM_KISS09>::init_time() const;

    template RNG<RAND_GPU_ALGORITHM_LCG12864>::RNG();
    template RNG<RAND_GPU_ALGORITHM_LCG12864>::RNG(size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_LCG12864>::RNG(unsigned long seed, size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_LCG12864>::~RNG();
    template size_t RNG<RAND_GPU_ALGORITHM_LCG12864>::buffer_size() const;
    template size_t RNG<RAND_GPU_ALGORITHM_LCG12864>::buffer_misses() const;
    template float RNG<RAND_GPU_ALGORITHM_LCG12864>::init_time() const;

    template RNG<RAND_GPU_ALGORITHM_LFIB>::RNG();
    template RNG<RAND_GPU_ALGORITHM_LFIB>::RNG(size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_LFIB>::RNG(unsigned long seed, size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_LFIB>::~RNG();
    template size_t RNG<RAND_GPU_ALGORITHM_LFIB>::buffer_size() const;
    template size_t RNG<RAND_GPU_ALGORITHM_LFIB>::buffer_misses() const;
    template float RNG<RAND_GPU_ALGORITHM_LFIB>::init_time() const;

    template RNG<RAND_GPU_ALGORITHM_MRG63K3A>::RNG();
    template RNG<RAND_GPU_ALGORITHM_MRG63K3A>::RNG(size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_MRG63K3A>::RNG(unsigned long seed, size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_MRG63K3A>::~RNG();
    template size_t RNG<RAND_GPU_ALGORITHM_MRG63K3A>::buffer_size() const;
    template size_t RNG<RAND_GPU_ALGORITHM_MRG63K3A>::buffer_misses() const;
    template float RNG<RAND_GPU_ALGORITHM_MRG63K3A>::init_time() const;

    template RNG<RAND_GPU_ALGORITHM_MSWS>::RNG();
    template RNG<RAND_GPU_ALGORITHM_MSWS>::RNG(size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_MSWS>::RNG(unsigned long seed, size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_MSWS>::~RNG();
    template size_t RNG<RAND_GPU_ALGORITHM_MSWS>::buffer_size() const;
    template size_t RNG<RAND_GPU_ALGORITHM_MSWS>::buffer_misses() const;
    template float RNG<RAND_GPU_ALGORITHM_MSWS>::init_time() const;

    template RNG<RAND_GPU_ALGORITHM_MT19937>::RNG();
    template RNG<RAND_GPU_ALGORITHM_MT19937>::RNG(size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_MT19937>::RNG(unsigned long seed, size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_MT19937>::~RNG();
    template size_t RNG<RAND_GPU_ALGORITHM_MT19937>::buffer_size() const;
    template size_t RNG<RAND_GPU_ALGORITHM_MT19937>::buffer_misses() const;
    template float RNG<RAND_GPU_ALGORITHM_MT19937>::init_time() const;

    template RNG<RAND_GPU_ALGORITHM_MWC64X>::RNG();
    template RNG<RAND_GPU_ALGORITHM_MWC64X>::RNG(size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_MWC64X>::RNG(unsigned long seed, size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_MWC64X>::~RNG();
    template size_t RNG<RAND_GPU_ALGORITHM_MWC64X>::buffer_size() const;
    template size_t RNG<RAND_GPU_ALGORITHM_MWC64X>::buffer_misses() const;
    template float RNG<RAND_GPU_ALGORITHM_MWC64X>::init_time() const;

    template RNG<RAND_GPU_ALGORITHM_PCG6432>::RNG();
    template RNG<RAND_GPU_ALGORITHM_PCG6432>::RNG(size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_PCG6432>::RNG(unsigned long seed, size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_PCG6432>::~RNG();
    template size_t RNG<RAND_GPU_ALGORITHM_PCG6432>::buffer_size() const;
    template size_t RNG<RAND_GPU_ALGORITHM_PCG6432>::buffer_misses() const;
    template float RNG<RAND_GPU_ALGORITHM_PCG6432>::init_time() const;

    template RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::RNG();
    template RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::RNG(size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::RNG(unsigned long seed, size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::~RNG();
    template size_t RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::buffer_size() const;
    template size_t RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::buffer_misses() const;
    template float RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::init_time() const;

    template RNG<RAND_GPU_ALGORITHM_RAN2>::RNG();
    template RNG<RAND_GPU_ALGORITHM_RAN2>::RNG(size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_RAN2>::RNG(unsigned long seed, size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_RAN2>::~RNG();
    template size_t RNG<RAND_GPU_ALGORITHM_RAN2>::buffer_size() const;
    template size_t RNG<RAND_GPU_ALGORITHM_RAN2>::buffer_misses() const;
    template float RNG<RAND_GPU_ALGORITHM_RAN2>::init_time() const;

    template RNG<RAND_GPU_ALGORITHM_TINYMT64>::RNG();
    template RNG<RAND_GPU_ALGORITHM_TINYMT64>::RNG(size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_TINYMT64>::RNG(unsigned long seed, size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_TINYMT64>::~RNG();
    template size_t RNG<RAND_GPU_ALGORITHM_TINYMT64>::buffer_size() const;
    template size_t RNG<RAND_GPU_ALGORITHM_TINYMT64>::buffer_misses() const;
    template float RNG<RAND_GPU_ALGORITHM_TINYMT64>::init_time() const;

    template RNG<RAND_GPU_ALGORITHM_TYCHE>::RNG();
    template RNG<RAND_GPU_ALGORITHM_TYCHE>::RNG(size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_TYCHE>::RNG(unsigned long seed, size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_TYCHE>::~RNG();
    template size_t RNG<RAND_GPU_ALGORITHM_TYCHE>::buffer_size() const;
    template size_t RNG<RAND_GPU_ALGORITHM_TYCHE>::buffer_misses() const;
    template float RNG<RAND_GPU_ALGORITHM_TYCHE>::init_time() const;

    template RNG<RAND_GPU_ALGORITHM_TYCHE_I>::RNG();
    template RNG<RAND_GPU_ALGORITHM_TYCHE_I>::RNG(size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_TYCHE_I>::RNG(unsigned long seed, size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_TYCHE_I>::~RNG();
    template size_t RNG<RAND_GPU_ALGORITHM_TYCHE_I>::buffer_size() const;
    template size_t RNG<RAND_GPU_ALGORITHM_TYCHE_I>::buffer_misses() const;
    template float RNG<RAND_GPU_ALGORITHM_TYCHE_I>::init_time() const;

    template RNG<RAND_GPU_ALGORITHM_WELL512>::RNG();
    template RNG<RAND_GPU_ALGORITHM_WELL512>::RNG(size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_WELL512>::RNG(unsigned long seed, size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_WELL512>::~RNG();
    template size_t RNG<RAND_GPU_ALGORITHM_WELL512>::buffer_size() const;
    template size_t RNG<RAND_GPU_ALGORITHM_WELL512>::buffer_misses() const;
    template float RNG<RAND_GPU_ALGORITHM_WELL512>::init_time() const;

    template RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::RNG();
    template RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::RNG(size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::RNG(unsigned long seed, size_t n_buffers, size_t multi);
    template RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::~RNG();
    template size_t RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::buffer_size() const;
    template size_t RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::buffer_misses() const;
    template float RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::init_time() const;


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
