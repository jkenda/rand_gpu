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

#include "RNG.hpp"
#include "rand_gpu.h"

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
#include "../include/cl.hpp"
#include "../kernel.hpp"


#define STATE_SIZE_KISS09 (4 * sizeof(cl_ulong))
#define STATE_SIZE_LCG12864 (2 * sizeof(cl_ulong))
#define STATE_SIZE_LFIB (17 * sizeof(cl_ulong) + 2 * sizeof(cl_char))
#define STATE_SIZE_MRG63K3A (6 * sizeof(cl_long))
#define STATE_SIZE_MSWS (max(sizeof(cl_ulong) + 2 * sizeof(cl_uint2), sizeof(ulong)))
#define STATE_SIZE_MT19937 (624 * sizeof(cl_uint) + sizeof(cl_int))
#define STATE_SIZE_MWC64X (max(sizeof(cl_ulong), 2 * sizeof(cl_uint)))
#define STATE_SIZE_PCG6432 (sizeof(cl_ulong))
#define STATE_SIZE_PHILOX2X32_10 (max(sizeof(cl_ulong), sizeof(cl_uint)))
#define STATE_SIZE_RAN2 (35 * sizeof(cl_int))
#define STATE_SIZE_TINYMT64 (3 * sizeof(cl_ulong) + 2 * sizeof(cl_uint))
#define STATE_SIZE_TYCHE (max(4 * sizeof(cl_uint), sizeof(cl_ulong)))
#define STATE_SIZE_TYCHE_I (max(4 * sizeof(cl_uint), sizeof(cl_ulong)))
#define STATE_SIZE_WELL512 (17 * sizeof(cl_uint))
#define STATE_SIZE_XORSHIFT6432STAR (sizeof(cl_ulong))

#define FLOAT_MULTI (1.0 / UINT32_MAX)
#define FLOAT_MULTI_MRG63K3A 1.0842021724855051562312e-19f
#define FLOAT_MULTI_MT19937 2.3283064365386962890625e-10f
#define FLOAT_MULTI_MWC64X 2.3283064365386963e-10f
#define FLOAT_MULTI_PCG6432 2.3283064365386963e-10f
#define FLOAT_MULTI_RAN2 4.6566128752457969230960e-10f
#define FLOAT_MULTI_WELL512 2.3283064365386963e-10f
#define FLOAT_MULTI_XORSHIFT6432STAR 2.3283064365386963e-10f

#define DOUBLE_MULTI (1.0 / UINT64_MAX)
#define DOUBLE_MULTI_MRG63K3A 1.0842021724855051562311819e-19
#define DOUBLE_MULTI_RAN2 2.1684043469904927853807e-19

using namespace std;
using chrono::duration_cast, chrono::nanoseconds, chrono::system_clock;

struct Buffer
{
    vector<uint8_t> _data;
    bool ready = true;
    mutex ready_lock;
    cl::Event ready_event;
    condition_variable ready_cond;

    inline void resize(const size_t size) { _data.resize(size); }
    inline uint8_t &operator[](const size_t i) { return _data[i]; }
    inline uint8_t *data() { return _data.data(); }
};


mutex constructor_lock;
mutex init_lock;
mutex queue_lock;

vector<cl::Device> devices;
size_t device_i = 0;
atomic<size_t> mem_all = 0;
bool initialized = false;

mt19937_64 generator;
cl::Context context;
cl::Device  device;
cl::Program program;

cl::NDRange _global_size;
size_t _buffer_size;
size_t _buf_limit;


struct RNG_private
{
    cl::CommandQueue queue;
    cl::Kernel k_generate;
    cl::Buffer state_buf;
    cl::Buffer device_buffer;

    const size_t _n_buffers;
    vector<Buffer> _host_buffers;
    uint_fast8_t active_buf = 0;

    size_t buf_offset = sizeof(long double);

    size_t _buffer_misses = 0;
    float _init_time;

    float _float_multi = FLOAT_MULTI;
    double _double_multi = DOUBLE_MULTI;

    RNG_private(size_t n_buffers = 2, size_t multi = 1, rand_gpu_algorithm algorithm = RAND_GPU_ALGORITHM_TYCHE, 
        bool use_custom_seed = false, const unsigned long custom_seed = 0)
    :
        _n_buffers(max(n_buffers, 2LU)),
        _host_buffers(max(n_buffers, 2LU))
    {
        cl_ulong seed;
        auto start = chrono::high_resolution_clock::now();

        {
            lock_guard<mutex> lock(constructor_lock);

            if (!initialized)
            {
                cl::Platform platform;

                // get platforms and devices
                try
                {
                    cl::Platform::get(&platform);
                    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
                    device = devices[device_i];
                }
                catch (const cl::Error& err)
                {
                    cerr << "No openCL platforms/devices found!\n";
                    throw err;
                }

                // create context
                context = cl::Context(devices);

                // build program
                cl::Program::Sources sources(1, make_pair(KERNEL_SOURCE, strlen(KERNEL_SOURCE)));
                program = cl::Program(context, sources);

                try
                {
                    program.build(devices);
                }
                catch (const cl::Error& err)
                {
                    // print buildlog if build failed
                    for (const cl::Device& dev : devices) 
                    {
                        if (program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev) != CL_SUCCESS)
                        {
                            string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                            cerr << buildlog << '\n';
                            throw err;
                        }
                    }
                }

                // initialize host RNG for generating seeds
                if (use_custom_seed)
                    generator = mt19937_64(system_clock::now().time_since_epoch().count());

                // get device info
                uint32_t max_cu = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
                size_t max_wg_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

                _global_size = cl::NDRange(max(multi, 1LU) * max_cu * max_wg_size);
                _buffer_size = _global_size[0] * sizeof(cl_ulong);
                _buf_limit = _buffer_size - sizeof(long double);

                initialized = true;
            }

            // generate seed
            if (use_custom_seed)
                seed = custom_seed;
            else
                seed = generator();
            
            // distribute devices among instances
            device = devices[device_i];
            device_i = (device_i + 1) % devices.size();

            // create command queue
            queue = cl::CommandQueue(context, device);
        }

        // increase total memory usage counter
        mem_all += _n_buffers * _buffer_size;

        // differentiate algorithms
        const char *name_init = "", *name_generate = "";
        size_t state_size = STATE_SIZE_MT19937;

        switch (algorithm)
        {
        case RAND_GPU_ALGORITHM_KISS09:
            name_init = "kiss09_init";
            name_generate = "kiss09_generate";
            state_size = STATE_SIZE_KISS09;
            break;
        case RAND_GPU_ALGORITHM_LCG12864:
            name_init = "lcg12864_init";
            name_generate = "lcg12864_generate";
            state_size = STATE_SIZE_LCG12864;
            break;
        case RAND_GPU_ALGORITHM_LFIB:
            name_init = "lfib_init";
            name_generate = "lfib_generate";
            state_size = STATE_SIZE_LFIB;
            break;
        case RAND_GPU_ALGORITHM_MRG63K3A:
            name_init = "mrg63k3a_init";
            name_generate = "mrg63k3a_generate";
            state_size = STATE_SIZE_MRG63K3A;
            break;
        case RAND_GPU_ALGORITHM_MSWS:
            name_init = "msws_init";
            name_generate = "msws_generate";
            state_size = STATE_SIZE_MSWS;
            break;
        case RAND_GPU_ALGORITHM_MT19937:
            name_init = "mt19937_init";
            name_generate = "mt19937_generate";
            state_size = STATE_SIZE_MT19937;
            _float_multi = FLOAT_MULTI_MT19937;
            break;
        case RAND_GPU_ALGORITHM_MWC64X:
            name_init = "mwc64x_init";
            name_generate = "mwc64x_generate";
            state_size = STATE_SIZE_MWC64X;
            _float_multi = FLOAT_MULTI_MWC64X;
            break;
        case RAND_GPU_ALGORITHM_PCG6432:
            name_init = "pcg6432_init";
            name_generate = "pcg6432_generate";
            state_size = STATE_SIZE_PCG6432;
            _float_multi = FLOAT_MULTI_PCG6432;
            break;
        case RAND_GPU_ALGORITHM_PHILOX2X32_10:
            name_init = "philox2x32_10_init";
            name_generate = "philox2x32_10_generate";
            state_size = STATE_SIZE_PHILOX2X32_10;
            break;
        case RAND_GPU_ALGORITHM_RAN2:
            name_init = "ran2_init";
            name_generate = "ran2_generate";
            state_size = STATE_SIZE_RAN2;
            break;
        case RAND_GPU_ALGORITHM_TINYMT64:
            name_init = "tinymt64_init";
            name_generate = "tinymt64_generate";
            state_size = STATE_SIZE_TINYMT64;
            break;
        case RAND_GPU_ALGORITHM_TYCHE_I:
            name_init = "tyche_i_init";
            name_generate = "tyche_i_generate";
            state_size = STATE_SIZE_TYCHE_I;
            break;
        case RAND_GPU_ALGORITHM_TYCHE:
            name_init = "tyche_init";
            name_generate = "tyche_generate";
            state_size = STATE_SIZE_TYCHE;
            break;
        case RAND_GPU_ALGORITHM_WELL512:
            name_init = "well512_init";
            name_generate = "well512_generate";
            state_size = STATE_SIZE_WELL512;
            _float_multi = FLOAT_MULTI_WELL512;
            break;
        case RAND_GPU_ALGORITHM_XORSHIFT6432STAR:
            name_init = "xorshift6432star_init";
            name_generate = "xorshift6432star_generate";
            state_size = STATE_SIZE_XORSHIFT6432STAR;
            _float_multi = FLOAT_MULTI_XORSHIFT6432STAR;
            break;
        }

        // resize host buffers, create device buffers
        state_buf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, _global_size[0] * state_size);
        device_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, _buffer_size);
        vector<cl::Buffer> temp_buffers(_n_buffers-1, cl::Buffer(context, CL_MEM_WRITE_ONLY, _buffer_size));
        for (Buffer &buf : _host_buffers)
        {
            buf.resize(_buffer_size);
        }

        // initialize RNG
        cl::Kernel k_init(program, name_init);
        cl::Event initialized;
        k_init.setArg(0, state_buf);

        switch (algorithm)
        {
        case RAND_GPU_ALGORITHM_TYCHE:
        case RAND_GPU_ALGORITHM_TYCHE_I:
            k_init.setArg(1, sizeof(cl_ulong), &seed);
            queue.enqueueNDRangeKernel(k_init, 0, _global_size);
            break;
        default:
            {
                ranlux48 s(seed);
                vector<uint64_t> seeds;
                seeds.reserve(_global_size[0]);
                for (size_t i = 0; i < _global_size[0]; i++)
                {
                    seeds.emplace_back(s());
                }
                cl::Buffer cl_seeds(context, seeds.begin(), seeds.end(), true, true);
                k_init.setArg(1, cl_seeds);
                queue.enqueueNDRangeKernel(k_init, cl::NullRange, _global_size[0]);
            }
        }

        // create kernel
        k_generate = cl::Kernel(program, name_generate);
        k_generate.setArg(0, state_buf);

        // fill all buffers
        vector<cl::Event> buffers_read(_n_buffers);

        for (size_t i = 0; i < _n_buffers-1; i++)
        {
            k_generate.setArg(1, temp_buffers[i]);
            queue.enqueueNDRangeKernel(k_generate, 0, _global_size);
            queue.enqueueReadBuffer(temp_buffers[i], false, 0, _buffer_size, _host_buffers[i].data(), nullptr, &buffers_read[i]);
        }
        k_generate.setArg(1, device_buffer);
        queue.enqueueNDRangeKernel(k_generate, 0, _global_size);
        queue.enqueueReadBuffer(device_buffer, false, 0, _buffer_size, _host_buffers[_n_buffers-1].data(), nullptr, &buffers_read[_n_buffers-1]);

        cl::Event::waitForEvents(buffers_read);

        // generate future numbers
        k_generate.setArg(1, device_buffer);
        queue.enqueueNDRangeKernel(k_generate, 0, _global_size);

        auto end = chrono::high_resolution_clock::now();
        _init_time = (duration_cast<nanoseconds>(end - start).count() / (float) 1'000'000);
    }

    ~RNG_private()
    {
        mem_all -= _n_buffers * _buffer_size;
    }


    template <typename T>
    T get_random()
    {
        Buffer &active_host_buf = _host_buffers[active_buf];

        // just switched buffers - wait for buffer to be ready
        if (buf_offset == 0)
        {
            unique_lock<mutex> lock(active_host_buf.ready_lock);
            if (!active_host_buf.ready) _buffer_misses++;
            active_host_buf.ready_cond.wait(lock, [&] { return active_host_buf.ready; });
        }

        // retrieve number from buffer
        T num;
        memcpy(&num, &active_host_buf[buf_offset], sizeof(T));
        buf_offset += sizeof(T);

        // out of numbers in current buffer
        if (buf_offset >= _buf_limit)
        {
            active_host_buf.ready = false;

            // enqueue reading data, generating future numbers
            queue.enqueueReadBuffer(device_buffer, false, 0, _buffer_size, active_host_buf.data(), nullptr, &active_host_buf.ready_event);
            queue.enqueueNDRangeKernel(k_generate, 0, _global_size);
            active_host_buf.ready_event.setCallback(CL_COMPLETE, set_flag, &active_host_buf);

            // switch active buffer
            active_buf = (active_buf + 1) % _n_buffers;
            buf_offset = 0;
        }
        
        return num;
    }

    size_t buffer_size() const
    {
        return _buffer_size;
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
    return mem_all;
}


/*
template specialization
*/

template <>
float RNG_private::get_random<float>()
{
    return _float_multi * get_random<uint32_t>();
}

template <>
double RNG_private::get_random<double>()
{
    return _double_multi * get_random<uint64_t>();
}

template <>
long double RNG_private::get_random<long double>()
{
    return _double_multi * get_random<uint64_t>();
}

template <>
bool RNG_private::get_random<bool>()
{
    return get_random<uint32_t>() > UINT32_MAX / 2 ? true : false;
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

size_t rand_gpu_buffer_size(rand_gpu_rng *rng) { return ((RNG_private *) rng)->buffer_size(); }
size_t rand_gpu_buf_misses(rand_gpu_rng *rng) { return ((RNG_private *) rng)->buffer_misses(); }
float rand_gpu_init_time(rand_gpu_rng *rng) { return ((RNG_private *) rng)->init_time(); }
size_t rand_gpu_memory() { return rand_gpu::memory_usage(); }


unsigned long  rand_gpu_u64(rand_gpu_rng *rng) { return ((RNG_private *) rng)->get_random<unsigned long>(); }
unsigned int   rand_gpu_u32(rand_gpu_rng *rng) { return ((RNG_private *) rng)->get_random<unsigned int>(); }
unsigned short rand_gpu_u16(rand_gpu_rng *rng) { return ((RNG_private *) rng)->get_random<unsigned short>(); }
unsigned char  rand_gpu_u8(rand_gpu_rng *rng)  { return ((RNG_private *) rng)->get_random<unsigned char>();  }

float       rand_gpu_float(rand_gpu_rng *rng)       { return ((RNG_private *) rng)->get_random<float>();       }
double      rand_gpu_double(rand_gpu_rng *rng)      { return ((RNG_private *) rng)->get_random<double>();      }
long double rand_gpu_long_double(rand_gpu_rng *rng) { return ((RNG_private *) rng)->get_random<long double>(); }

const char *rand_gpu_algorithm_name(rand_gpu_algorithm algorithm, bool long_name) { return rand_gpu::algorithm_name(algorithm, long_name); }

}


/*
RNG definitions
*/

namespace rand_gpu
{
    RNG::RNG(rand_gpu_algorithm algorithm, size_t n_buffers, size_t multi)
    :   
        d_ptr_(make_unique<RNG_private>(n_buffers, multi, algorithm, false))
    {
    }

    RNG::RNG(unsigned long seed, rand_gpu_algorithm algorithm, size_t n_buffers, size_t multi)
    :   
        d_ptr_(make_unique<RNG_private>(n_buffers, multi, algorithm, true, seed))
    {
    }

    RNG::RNG(RNG&&) = default;
    RNG& RNG::operator=(RNG&&) = default;

    RNG::~RNG() = default;

    template <typename T>
    T RNG::get_random()
    {
        return d_ptr_->get_random<T>();
    }

    /*
    instantiate templates for all primitives
    */

    template unsigned long long int RNG::get_random<unsigned long long int>();
    template unsigned long int  RNG::get_random<unsigned long int>();
    template unsigned int       RNG::get_random<unsigned int>();
    template unsigned short int RNG::get_random<unsigned short int>();
    template unsigned char      RNG::get_random<unsigned char>();

    template long long int RNG::get_random<long long int>();
    template long int      RNG::get_random<long int>();
    template int           RNG::get_random<int>();
    template short int     RNG::get_random<short int>();
    template signed char   RNG::get_random<signed char>();

    template char RNG::get_random<char>();
    template bool RNG::get_random<bool>();

    template float       RNG::get_random<float>();
    template double      RNG::get_random<double>();
    template long double RNG::get_random<long double>();

    size_t RNG::buffer_size() const
    {
        return d_ptr_->buffer_size();
    }

    size_t RNG::buffer_misses() const
    {
        return d_ptr_->buffer_misses();
    }

    float RNG::init_time() const
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

}
