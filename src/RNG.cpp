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
#include <mutex>
#include <condition_variable>
#include <array>
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

/*
TODO: queue on graphics card
*/

using namespace std;
using chrono::duration_cast, chrono::microseconds, chrono::system_clock;

struct Buffer
{
    vector<uint8_t> _data;
    bool ready;
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

cl::NDRange global_size;
size_t buf_size;
size_t buf_limit;


class RNG_private
{
    cl::CommandQueue queue;
    cl::Kernel k_generate;
    cl::Buffer state_buf;
    cl::Buffer device_buffer;

    array<Buffer, 2> host_buffer;
    uint_fast8_t active_buf = 0;

    size_t buf_offset = sizeof(long double);

public:
    RNG_private(const size_t multi = 1, const unsigned long custom_seed = 0)
    {
        cl_ulong seed;

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
                generator = mt19937_64(duration_cast<microseconds>(system_clock::now().time_since_epoch()).count());

                // get device info
                uint32_t max_cu = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
                size_t max_wg_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

                global_size = cl::NDRange(max(multi, 1LU) * max_cu * max_wg_size);
                buf_size = global_size[0] * sizeof(cl_ulong);
                buf_limit = buf_size - sizeof(long double);

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
        mem_all += 2 * buf_size;

        // resize host buffers, create device buffers
        host_buffer[0].resize(buf_size);
        host_buffer[1].resize(buf_size);
        state_buf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, global_size[0] * TYCHE_I_STATE_SIZE);
        device_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, buf_size);
        cl::Buffer device_buffer_temp(context, CL_MEM_WRITE_ONLY, buf_size);

        // initialize RNG
        cl::Kernel k_init(program, "init");
        k_init.setArg(0, state_buf);
        k_init.setArg(1, sizeof(cl_ulong), &seed);
        queue.enqueueNDRangeKernel(k_init, 0, global_size);

        // create kernel
        k_generate = cl::Kernel(program, "generate");
        k_generate.setArg(0, state_buf);

        // fill both buffers
        vector<cl::Event> buffers_read(2);
        k_generate.setArg(1, device_buffer_temp);
        queue.enqueueNDRangeKernel(k_generate, 0, global_size);
        k_generate.setArg(1, device_buffer);
        queue.enqueueNDRangeKernel(k_generate, 0, global_size);
        queue.enqueueReadBuffer(device_buffer, false, 0, buf_size, host_buffer[0].data(), nullptr, &buffers_read[0]);
        queue.enqueueReadBuffer(device_buffer_temp, false, 0, buf_size, host_buffer[1].data(), nullptr, &buffers_read[1]);
        cl::Event::waitForEvents(buffers_read);
        host_buffer[0].ready = true;
        host_buffer[1].ready = true;

        // generate future numbers
        queue.enqueueNDRangeKernel(k_generate, 0, global_size);
    }

    ~RNG_private()
    {
        mem_all -= 2 * buf_size;
    }


    template <typename T>
    T get_random()
    {
        Buffer &active_host_buf = host_buffer[active_buf];

        // just switched buffers - wait for buffer to be ready
        if (buf_offset == 0)
        {
            unique_lock<mutex> lock(active_host_buf.ready_lock);
            active_host_buf.ready_cond.wait(lock, [&] { return active_host_buf.ready; });
        }

        // retrieve number from buffer
        T num;
        memcpy(&num, &active_host_buf[buf_offset], sizeof(T));
        buf_offset += sizeof(T);

        // out of numbers in current buffer
        if (buf_offset >= buf_limit)
        {
            active_host_buf.ready = false;

            // enqueue reading data, generating future numbers
            queue.enqueueReadBuffer(device_buffer, false, 0, buf_size, active_host_buf.data(), nullptr, &active_host_buf.ready_event);
            queue.enqueueNDRangeKernel(k_generate, 0, global_size);
            active_host_buf.ready_event.setCallback(CL_COMPLETE, set_flag, &active_host_buf);

            // switch active buffer
            active_buf = !active_buf;
            buf_offset = 0;
        }
        return num;
    }

    size_t buffer_size() const
    {
        return buf_size;
    }

private:
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

rand_gpu_rng *rand_gpu_new(const size_t multi)
{
    return (rand_gpu_rng *) new RNG_private(multi, 0);
}

rand_gpu_rng *rand_gpu_new_with_seed(const size_t multi, const unsigned long seed)
{
    return (rand_gpu_rng *) new RNG_private(multi, seed);
}

void rand_gpu_delete(rand_gpu_rng *rng)
{
    delete (RNG_private *) rng;
}

size_t rand_gpu_buffer_size(rand_gpu_rng *rng) { return ((RNG_private *) rng)->buffer_size(); }
size_t rand_gpu_memory() { return mem_all; }

unsigned long  rand_gpu_u64(rand_gpu_rng *rng) { return ((RNG_private *) rng)->get_random<unsigned long>(); }
unsigned int   rand_gpu_u32(rand_gpu_rng *rng) { return ((RNG_private *) rng)->get_random<unsigned int>(); }
unsigned short rand_gpu_u16(rand_gpu_rng *rng) { return ((RNG_private *) rng)->get_random<unsigned short>(); }
unsigned char  rand_gpu_u8(rand_gpu_rng *rng)  { return ((RNG_private *) rng)->get_random<unsigned char>();  }

float       rand_gpu_float(rand_gpu_rng *rng)       { return ((RNG_private *) rng)->get_random<float>();       }
double      rand_gpu_double(rand_gpu_rng *rng)      { return ((RNG_private *) rng)->get_random<double>();      }
long double rand_gpu_long_double(rand_gpu_rng *rng) { return ((RNG_private *) rng)->get_random<long double>(); }

}


/*
RNG definitions
*/

namespace rand_gpu
{
    RNG::RNG(size_t multi, unsigned long seed)
    :   d_ptr_(make_unique<RNG_private>(multi, seed))
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

    template bool      RNG::get_random<bool>();

    template float       RNG::get_random<float>();
    template double      RNG::get_random<double>();
    template long double RNG::get_random<long double>();

    size_t RNG::buffer_size() const
    {
        return d_ptr_->buffer_size();
    }

    size_t memory_usage()
    {
        return mem_usage();
    }

}
