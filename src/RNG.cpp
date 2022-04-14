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

#include <thread>
#include <cstdio>
#include <random>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstring>
#include "../kernel.hpp"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#include "../include/cl.hpp"


#define TYCHE_I_STATE_SIZE (4 * sizeof(cl_uint))

#define TYCHE_I_FLOAT_MULTI 5.4210108624275221700372640e-20f
#define TYCHE_I_DOUBLE_MULTI 5.4210108624275221700372640e-20

/*
FIXME: /usr/bin/ld: /tmp/cczIhVEY.o: undefined reference to symbol 'clReleaseCommandQueue@@OPENCL_1.0'
       when linking to C++ file

TODO: circular buffer on graphics card, offset read
*/

using namespace std;

namespace rand_gpu
{
    static mutex constructor_lock;
    static mutex init_lock;
    static mutex queue_lock;

    static bool initialized = false;
    static size_t mem_all = 0L;

    static cl::Context context;
    static cl::Device  device;
    static cl::Program program;

    static size_t nthreads;
    static size_t wg_size;
    static size_t buf_size;
    static size_t buf_limit;


    RNG::RNG(size_t multi)
    :
        active(0), buffer_i(sizeof(uint32_t))
    {
        constructor_lock.lock();

        if (!initialized) {
            cl::Platform platform;
            vector<cl::Device> devices;

            // get platforms and devices
            cl::Platform::get(&platform);
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            device = devices.at(0);

            // get device info
            uint32_t max_cu = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
            uint32_t max_wg_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
            wg_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
            nthreads = multi * max_cu * max_wg_size;
            buf_size = nthreads * sizeof(cl_ulong);
            buf_limit = buf_size - sizeof(long double);

            // create context
            context = cl::Context({ device });

            // build program
            cl::Program::Sources sources(1, make_pair(KERNEL_SOURCE, strlen(KERNEL_SOURCE)));
            program = cl::Program(context, sources);

            try {
                program.build({ device }, "");
            }
            catch (cl::Error) {
                std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                fputs(buildlog.c_str(), stderr);
                throw cl::Error(CL_BUILD_PROGRAM_FAILURE);
            }

            initialized = true;
        }

        mem_all += buf_size;

        constructor_lock.unlock();

        cl::Event e;

        // create command queue
        queue = cl::CommandQueue(context, device);

        // generate seeds
        vector<cl_ulong> seeds(nthreads);
        random_device rd;
        mt19937_64 generator(rd());
        for (cl_ulong &seed : seeds) {
            seed = generator();
        }
        cl::Buffer seed_buffer(context, seeds.begin(), seeds.end(), true);
        state_buf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, nthreads * TYCHE_I_STATE_SIZE);

        // initialize RNG
        cl::Kernel k_init = cl::Kernel(program, "init");
        k_init.setArg(0, state_buf);
        k_init.setArg(1, seed_buffer);
        queue.enqueueNDRangeKernel(k_init, cl::NDRange(0), cl::NDRange(nthreads), cl::NullRange, NULL, &e);
        e.wait();

        // create kernel
        k_generate = cl::Kernel(program, "generate");

        // resize buffers
        buffer[0].data.resize(buf_size);
        buffer[1].data.resize(buf_size);

        // create buffer
        random_buf = cl::Buffer(context, CL_MEM_WRITE_ONLY, buf_size, NULL);

        // fill both buffers
        k_generate.setArg(0, state_buf);
        k_generate.setArg(1, random_buf);
        queue.enqueueNDRangeKernel(k_generate, 0, cl::NDRange(nthreads), cl::NullRange, NULL, &e);
        e.wait();
        queue.enqueueReadBuffer(random_buf, CL_TRUE, 0, buf_size, buffer[0].data.data());
        queue.enqueueNDRangeKernel(k_generate, 0, cl::NDRange(nthreads), cl::NullRange, NULL, &e);
        e.wait();
        queue.enqueueReadBuffer(random_buf, CL_TRUE, 0, buf_size, buffer[1].data.data());
        buffer[0].ready = true;
        buffer[1].ready = true;

        // generate future numbers
        queue.enqueueNDRangeKernel(k_generate, 0, cl::NDRange(nthreads), cl::NullRange);
    }

    void RNG::set_flag(cl_event e, cl_int status, void *data)
    {
        RNG *inst = static_cast<RNG *>(data);
        lock_guard<mutex> lock(inst->buffer_ready_lock);
        inst->buffer[!inst->active].ready = true;
        inst->buffer_ready_cond.notify_one();
    }

    template <typename T>
    T RNG::rand()
    {
        Buffer &active_buffer = buffer[active];

        if (buffer_i == 0) {
            // check if buffer is ready
            unique_lock<mutex> lock(buffer_ready_lock);
            active = !active;
            buffer_ready_cond.wait(lock, [&] { return active_buffer.ready; });
            active = !active;
        }

        T num;
        memcpy(&num, &active_buffer.data[buffer_i], sizeof(T));
        buffer_i += sizeof(T);

        // out of numbers in current buffer
        if (buffer_i >= buf_limit) {
            active_buffer.ready = false;

            /*
            OpenCL API calls that queue commands to a command-queue or change the state of OpenCL objects 
            such as command-queue objects, memory objects, program and kernel objects are not thread-safe.
            */

            // enqueue read data into the empty buffer, generate future numbers
            {
                lock_guard<mutex> lock(queue_lock);
                queue.enqueueReadBuffer(random_buf, CL_FALSE, 0, buf_size, active_buffer.data.data(), NULL, &active_buffer.ready_event);
                queue.enqueueNDRangeKernel(k_generate, 0, cl::NDRange(nthreads), cl::NullRange);
                active_buffer.ready_event.setCallback(CL_COMPLETE, set_flag, this);
            }

            // switch active buffer
            active = !active;
            buffer_i = 0;
        }
        return num;
    }

    size_t RNG::buffer_size()
    {
        return 2 * buf_size;
    }

    RNG::~RNG()
    {
        constructor_lock.lock();
        mem_all -= buf_size;
        constructor_lock.unlock();
        queue.flush();
        queue.finish();
    }

    /*
    template specialization
    */

    template <>
    float RNG::rand<float>()
    {
        return rand<uint32_t>() / (float) UINT32_MAX;
    }

    template <>
    double RNG::rand<double>()
    {
        return rand<uint64_t>() / (double) UINT64_MAX;
    }

    template <>
    long double RNG::rand<long double>()
    {
        return rand<uint64_t>() / (long double) UINT64_MAX;
    }

    template <>
    bool RNG::rand<bool>()
    {
        return rand<uint64_t>() > UINT64_MAX / 2 ? true : false;
    }

    /*
    instantiate templates for all primitives
    */

    template uint64_t RNG::rand<uint64_t>();
    template uint32_t RNG::rand<uint32_t>();
    template uint16_t RNG::rand<uint16_t>();
    template uint8_t RNG::rand<uint8_t>();
    template int64_t RNG::rand<int64_t>();
    template int32_t RNG::rand<int32_t>();
    template int16_t RNG::rand<int16_t>();
    template int8_t RNG::rand<int8_t>();

#ifdef __uint128_t
    template __uint128_t RNG::rand<__uint128_t>();
#endif

} // namespace rand_gpu


/*
C function wrappers
*/

vector<rand_gpu::RNG *> RNGs;

extern "C" {

int rand_gpu_new(uint32_t multi)
{
    lock_guard<mutex> lock(rand_gpu::init_lock);
    RNGs.emplace_back(new rand_gpu::RNG(multi));
    return RNGs.size() - 1;
}

void rand_gpu_delete(int rng)
{
    lock_guard<mutex> lock(rand_gpu::init_lock);
    delete RNGs[rng];
    RNGs.erase(RNGs.begin() + rng);
}

size_t rand_gpu_buffer_size(rand_gpu_rng rng) { return RNGs[rng]->buffer_size(); }
size_t rand_gpu_memory() { return rand_gpu::mem_all; }

uint64_t rand_gpu_u64(int rng) { return RNGs[rng]->rand<uint64_t>(); }
uint32_t rand_gpu_u32(int rng) { return RNGs[rng]->rand<uint32_t>(); }
uint16_t rand_gpu_u16(int rng) { return RNGs[rng]->rand<uint16_t>(); }
uint8_t  rand_gpu_u8(int rng)  { return RNGs[rng]->rand<uint8_t>();  }

float       rand_gpu_float(int rng)       { return RNGs[rng]->rand<float>();       }
double      rand_gpu_double(int rng)      { return RNGs[rng]->rand<double>();      }
long double rand_gpu_long_double(int rng) { return RNGs[rng]->rand<long double>(); }

}
