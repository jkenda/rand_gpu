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


#define TYCHE_I_STATE_SIZE (4 * sizeof(cl_uint))

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

    RNG_private(const size_t n_buffers = 2, const size_t multi = 1,
        rand_gpu_algorithm algorithm = RAND_GPU_ALGORITHM_TYCHE, const unsigned long custom_seed = 0)
    :
        _n_buffers(max(n_buffers, 2LU)),
        _host_buffers(_n_buffers)
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
                if (custom_seed == 0)
                    generator = mt19937_64(duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count());

                // get device info
                uint32_t max_cu = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
                size_t max_wg_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

                _global_size = cl::NDRange(max(multi, 1LU) * max_cu * max_wg_size);
                _buffer_size = _global_size[0] * sizeof(cl_ulong);
                _buf_limit = _buffer_size - sizeof(long double);

                initialized = true;
            }

            // generate seed
            if (custom_seed == 0)
                seed = generator();
            else
                seed = custom_seed;
            
            // distribute devices among instances
            device = devices[device_i];
            device_i = (device_i + 1) % devices.size();

            // create command queue
            queue = cl::CommandQueue(context, device);
        }

        // increase total memory usage counter
        mem_all += _n_buffers * _buffer_size;

        // resize host buffers, create device buffers
        state_buf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, _global_size[0] * TYCHE_I_STATE_SIZE);
        device_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, _buffer_size);
        vector<cl::Buffer> temp_buffers(_n_buffers-1, cl::Buffer(context, CL_MEM_WRITE_ONLY, _buffer_size));
        for (Buffer &buf : _host_buffers)
        {
            buf.resize(_buffer_size);
        }

        // differentiate algorithms
        const char *name_init = "", *name_generate = "";

        switch (algorithm)
        {
        case RAND_GPU_ALGORITHM_KISS09:
            name_init = "kiss09_init";
            name_generate = "kiss09_generate";
            break;
        case RAND_GPU_ALGORITHM_LCG12864:
            name_init = "lcg12864_init";
            name_generate = "lcg12864_generate";
            break;
        case RAND_GPU_ALGORITHM_LFIB:
            name_init = "lfib_init";
            name_generate = "lfib_generate";
            break;
        case RAND_GPU_ALGORITHM_MRG63K3A:
            name_init = "mrg63k3a_init";
            name_generate = "mrg63k3a_generate";
            break;
        case RAND_GPU_ALGORITHM_MSWS:
            name_init = "msws_init";
            name_generate = "msws_generate";
            break;
        case RAND_GPU_ALGORITHM_MT19937:
            name_init = "mt19937_init";
            name_generate = "mt19937_generate";
            break;
        case RAND_GPU_ALGORITHM_PHILOX2X32_10:
            name_init = "philox2x32_10_init";
            name_generate = "philox2x32_10_generate";
            break;
        case RAND_GPU_ALGORITHM_TINYMT64:
            name_init = "tinymt64_init";
            name_generate = "tinymt64_generate";
            break;
        case RAND_GPU_ALGORITHM_TYCHE_I:
            name_init = "tyche_i_init";
            name_generate = "tyche_i_generate";
            break;
        case RAND_GPU_ALGORITHM_TYCHE:
            name_init = "tyche_init";
            name_generate = "tyche_generate";
            break;
        }

        // initialize RNG
        cl::Kernel k_init(program, name_init);
        cl::Event initialized;
        k_init.setArg(0, state_buf);

        switch (algorithm)
        {
        case RAND_GPU_ALGORITHM_TYCHE:
        case RAND_GPU_ALGORITHM_TYCHE_I:
        {
            k_init.setArg(1, sizeof(cl_ulong), &seed);
            queue.enqueueNDRangeKernel(k_init, cl::NullRange, _global_size, cl::NullRange, nullptr, &initialized);
            break;
        }
        default:
        {
            mt19937_64 s(seed);
            vector<uint64_t> seeds;
            seeds.reserve(_global_size[0]);
            for (size_t i = 0; i < _global_size[0]; i++)
            {
                seeds.emplace_back(s());
            }
            cl::Buffer cl_seeds(context, seeds.begin(), seeds.end(), false);
            k_init.setArg(1, cl_seeds);
            queue.enqueueNDRangeKernel(k_init, cl::NullRange, _global_size, cl::NullRange, nullptr, &initialized);
            break;
        }
        }
        initialized.wait();

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
    return get_random<uint32_t>() / (float) UINT32_MAX;
}

template <>
double RNG_private::get_random<double>()
{
    return get_random<uint64_t>() / (double) UINT64_MAX;
}

template <>
long double RNG_private::get_random<long double>()
{
    return get_random<uint64_t>() / (long double) UINT64_MAX;
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

rand_gpu_rng *rand_gpu_new(size_t n_buffers, size_t buffer_multi, enum rand_gpu_algorithm algorithm)
{
    return (rand_gpu_rng *) new RNG_private(n_buffers, buffer_multi, algorithm);
}

rand_gpu_rng *rand_gpu_new_with_seed(size_t n_buffers, size_t buffer_multi, enum rand_gpu_algorithm algorithm, unsigned long seed)
{
    return (rand_gpu_rng *) new RNG_private(n_buffers, buffer_multi, algorithm, seed);
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

const char *rand_gpu_algorithm_name(rand_gpu_algorithm algorithm) { return rand_gpu::algorithm_name(algorithm); }

}


/*
RNG definitions
*/

namespace rand_gpu
{
    RNG::RNG(size_t n_buffers, size_t multi, rand_gpu_algorithm algorithm, unsigned long seed)
    :   
        d_ptr_(make_unique<RNG_private>(n_buffers, multi, algorithm, seed))
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

    const char *algorithm_name(rand_gpu_algorithm algorithm)
    {
        const char *names[] = {
            "Kiss09 - KISS (Keep It Simple, Stupid)",
            "LCG12864 - 128-bit Linear Congruential Generator",
            "LFib - Multiplicative Lagged Fibbonaci generator (lowest bit is always 1)",
            "MRG63K3A (Multiple Recursive Generator)",
            "MSWS - Middle Square Weyl Sequence",
            "MT19937 - Mersenne twister: a 623-dimensionally equidistributed uniform pseudo-random number generator",
            "Philox2x32_10",
            "Tinymt64 - Tiny mersenne twister",
            "Tyche - 512-bit Tyche (Well-Equidistributed Long-period Linear RNG)",
            "Tyche-i - a faster variant of Tyche with a shorter period",
        };
        return names[algorithm];
    }

}
