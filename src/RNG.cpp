/**
 * @file RNG.cpp
 * @author Jakob Kenda (kenda.jakob@gmail.com)
 * @brief Source file for RNG/rand_gpu C/C++ library
 * @version 0.2
 * @date 2022-04-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "RNG.hpp"
#include "rand_gpu.h"

#include <vector>
#include <mutex>
#include <condition_variable>
#include <array>
#include <memory>
#include <random>
#include <iostream>
#include <cstring>
#include <atomic>
#include "../kernel.hpp"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#include "../include/cl.hpp"


#define TYCHE_I_STATE_SIZE (4 * sizeof(cl_uint))

#define TYCHE_I_FLOAT_MULTI 5.4210108624275221700372640e-20f
#define TYCHE_I_DOUBLE_MULTI 5.4210108624275221700372640e-20

/*
TODO: circular buffer on graphics card, offset read
*/

using namespace std;

namespace rand_gpu
{
    struct Buffer
    {
        vector<uint8_t> data;
        cl::Event ready_event;
        bool ready;
    };

    struct waiting_ref
    {
        bool& ready;
        mutex& ready_lock;
        condition_variable& ready_cond;
    };

    mutex constructor_lock;
    mutex init_lock;
    mutex queue_lock;

    vector<cl::Device> devices;
    atomic<size_t> device_i = 0;
    atomic<size_t> mem_all = 0;
    bool initialized = false;

    mt19937_64 generator;
    cl::Context context;
    cl::Device  device;
    cl::Program program;

    cl::NDRange global_size;
    cl::NDRange local_size;
    size_t buf_size;
    size_t buf_limit;


    class RNG_private
    {
        mutex buffer_ready_lock;
        condition_variable buffer_ready_cond;

        cl::CommandQueue queue;

        cl::Buffer  state_buf;
        cl::Buffer  random_buf;
        cl::Kernel  k_generate;

        array<Buffer, 2> buffer;
        size_t buffer_max_size;
        uint_fast8_t active_buf;
        uint_fast8_t waiting_buf;

        size_t buf_offset;

    public:
        RNG_private(size_t multi)
        :
            active_buf(0), buf_offset(sizeof(uint32_t))
        {
            constructor_lock.lock();

            if (!initialized)
            {
                cl::Platform platform;

                // get platforms and devices
                try
                {
                    cl::Platform::get(&platform);
                    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
                    device = devices.at(device_i);
                }
                catch(const cl::Error& err)
                {
                    cerr << "No openCL devices found!\n";
                    throw err;
                }

                // create context
                context = cl::Context(devices);

                // build program
                cl::Program::Sources sources(1, make_pair(KERNEL_SOURCE, strlen(KERNEL_SOURCE)));
                program = cl::Program(context, sources);

                try
                {
                    program.build(devices, "");
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

                // initialize random number generator
                random_device rd;
                generator = mt19937_64(rd());

                size_t preferred_multiple;
                cl::Kernel kernel = cl::Kernel(program, "generate");
                kernel.getWorkGroupInfo(device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &preferred_multiple);

                // get device info
                uint32_t max_cu = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
                size_t mem_size = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
                size_t max_wg_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

                buffer_max_size = mem_size - mem_size % sizeof(cl_ulong);
                local_size = cl::NDRange(max_wg_size);
                global_size = cl::NDRange(multi * max_cu * local_size[0]);
                buf_size = global_size[0] * sizeof(cl_ulong);
                buf_limit = buf_size - sizeof(long double);

                initialized = true;
            }

            // generate seed
            cl_ulong seed = generator();

            constructor_lock.unlock();

            // increase total memory usage counter
            mem_all += 2 * buf_size;

            device = devices.at(device_i);
            device_i = (device_i + 1) % devices.size();

            // create command queue
            queue = cl::CommandQueue(context, device);

            // resize host buffers
            buffer[0].data.resize(buf_size);
            buffer[1].data.resize(buf_size);

            // create buffers
            state_buf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, global_size[0] * TYCHE_I_STATE_SIZE);
            random_buf = cl::Buffer(context, CL_MEM_WRITE_ONLY, buf_size);

            // initialize RNG
            cl::Kernel k_init = cl::Kernel(program, "init");
            k_init.setArg(0, state_buf);
            k_init.setArg(1, sizeof(cl_ulong), &seed);
            queue.enqueueNDRangeKernel(k_init, 0, global_size);

            // create kernel
            k_generate = cl::Kernel(program, "generate");

            // fill both buffers
            k_generate.setArg(0, state_buf);
            k_generate.setArg(1, random_buf);
            queue.enqueueNDRangeKernel(k_generate, 0, global_size);
            queue.enqueueReadBuffer(random_buf, CL_TRUE, 0, buf_size, buffer[0].data.data());
            queue.enqueueNDRangeKernel(k_generate, 0, global_size);
            queue.enqueueReadBuffer(random_buf, CL_TRUE, 0, buf_size, buffer[1].data.data());
            buffer[0].ready = true;
            buffer[1].ready = true;

            // generate future numbers
            queue.enqueueNDRangeKernel(k_generate, 0, global_size);
        }

        RNG_private(RNG_private&) = delete;
        RNG_private(RNG_private&&) = delete;
        RNG_private& operator=(RNG_private&) = delete;
        RNG_private& operator=(RNG_private&&) = delete;

        ~RNG_private()
        {
            mem_all -= 2 * buf_size;
        }


        template <typename T>
        T get_random()
        {
            Buffer &active_buffer = buffer[active_buf];

            // just switched buffers
            if (buf_offset == 0)
            {
                unique_lock<mutex> lock(buffer_ready_lock);
                buffer_ready_cond.wait(lock, [&] { return active_buffer.ready; });
            }

            // retrieve number from buffer
            T num;
            memcpy(&num, &active_buffer.data[buf_offset], sizeof(T));
            buf_offset += sizeof(T);

            // out of numbers in current buffer
            if (buf_offset >= buf_limit)
            {
                active_buffer.ready = false;

                // enqueue reading data, generating future numbers
                queue.enqueueReadBuffer(random_buf, CL_FALSE, 0, buf_size, active_buffer.data.data(), nullptr, &active_buffer.ready_event);
                queue.enqueueNDRangeKernel(k_generate, 0, global_size);
                active_buffer.ready_event.setCallback(CL_COMPLETE, set_flag, 
                        new waiting_ref{ active_buffer.ready, buffer_ready_lock, buffer_ready_cond });

                // switch active buffer
                active_buf = 1^active_buf;
                buf_offset = 0;
            }
            return num;
        }

        size_t buffer_size()
        {
            return buf_size;
        }

    private:
        static void set_flag(cl_event e, cl_int status, void *data)
        {
            waiting_ref *const ptr = (waiting_ref *) data;
            lock_guard<mutex> lock(ptr->ready_lock);
            ptr->ready = true;
            ptr->ready_cond.notify_one();
            delete ptr;
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

}

/*
C function wrappers
*/

using rand_gpu::RNG_private;

extern "C" {

rand_gpu_rng *rand_gpu_new(uint32_t multi)
{
    return (rand_gpu_rng *) new RNG_private(multi);
}

void rand_gpu_delete(rand_gpu_rng *rng)
{
    delete (RNG_private *) rng;
}

size_t rand_gpu_buffer_size(rand_gpu_rng *rng) { return ((RNG_private *) rng)->buffer_size(); }
size_t rand_gpu_memory() { return rand_gpu::mem_all; }

uint64_t rand_gpu_u64(rand_gpu_rng *rng) { return ((RNG_private *) rng)->get_random<uint64_t>(); }
uint32_t rand_gpu_u32(rand_gpu_rng *rng) { return ((RNG_private *) rng)->get_random<uint32_t>(); }
uint16_t rand_gpu_u16(rand_gpu_rng *rng) { return ((RNG_private *) rng)->get_random<uint16_t>(); }
uint8_t  rand_gpu_u8(rand_gpu_rng *rng)  { return ((RNG_private *) rng)->get_random<uint8_t>();  }

float       rand_gpu_float(rand_gpu_rng *rng)       { return ((RNG_private *) rng)->get_random<float>();       }
double      rand_gpu_double(rand_gpu_rng *rng)      { return ((RNG_private *) rng)->get_random<double>();      }
long double rand_gpu_long_double(rand_gpu_rng *rng) { return ((RNG_private *) rng)->get_random<long double>(); }

}

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

    template float       RNG::get_random<float>();
    template double      RNG::get_random<double>();
    template long double RNG::get_random<long double>();

    size_t RNG::buffer_size()
    {
        return d_ptr_->buffer_size();
    }

    size_t memory_usage()
    {
        return rand_gpu::mem_usage();
    }

} // namespace rand_gpu
