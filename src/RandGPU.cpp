/**
 * @file RandGPU.cpp
 * @author Jakob Kenda (kenda.jakob@gmail.com)
 * @brief Source file for RandGPU/rand_gpu C/C++ library
 *        NOTE: if you are getting include errors, run "make kernel.hpp"
 * @version 0.1
 * @date 2022-04-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "RandGPU.hpp"

#include <thread>
#include <cstdio>
#include <random>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstring>
#include "exceptions.hpp"
#include "../kernel.hpp"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#include "../include/cl.hpp"

#define TYCHE_I_STATE_SIZE (4 * sizeof(cl_uint))

using namespace std;

#define TYCHE_I_FLOAT_MULTI 5.4210108624275221700372640e-20f
#define TYCHE_I_DOUBLE_MULTI 5.4210108624275221700372640e-20

/*
FIXME: /usr/bin/ld: /tmp/cczIhVEY.o: undefined reference to symbol 'clReleaseCommandQueue@@OPENCL_1.0'
       when linking to C++ file

TODO: circular buffer on graphics card, offset read
*/

static mutex constructor_lock;
static mutex init_lock;
static mutex buffer_ready_lock;
static condition_variable buffer_ready_cond;


static bool initialized = false;
static size_t mem_all = 0L;

static cl::Context context;
static cl::Device  device;
static cl::Program program;
static cl::Buffer  state_buf;

static size_t nthreads;
static size_t wg_size;
static size_t buf_size;
static size_t buf_limit;


RandGPU::RandGPU(size_t multi)
:
    active(0), buffer_i(sizeof(uint32_t))
{
    constructor_lock.lock();

    cl::Event e;

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
    }

    // create command queue
    queue = cl::CommandQueue(context, device);

    if (!initialized) {

        // build program
        cl::Program::Sources sources(1, make_pair(KERNEL_SOURCE.c_str(), KERNEL_SOURCE.length()));
        program = cl::Program(context, sources);

        try {
            program.build({ device }, "");
        }
        catch (cl::Error) {
            std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            fputs(buildlog.c_str(), stderr);
            throw cl::Error(CL_BUILD_PROGRAM_FAILURE);
        }

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

        initialized = true;
    }

    // create kernel
    k_generate = cl::Kernel(program, "generate");

    // resize buffers
    buffer[0].data.resize(buf_size);
    buffer[1].data.resize(buf_size);
    mem_all += 2 * buf_size;

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

    constructor_lock.unlock();
}

void set_flag(cl_event e, cl_int status, void *data)
{
    lock_guard<mutex> lock(buffer_ready_lock);
	*(bool *) data = true;
    buffer_ready_cond.notify_one();
}

template <typename R>
R RandGPU::rand()
{
    Buffer &active_buffer = buffer[active];

    if (buffer_i == 0) {
		// check if buffer is ready
		unique_lock<mutex> lock(buffer_ready_lock);
        buffer_ready_cond.wait(lock, [&] { return active_buffer.ready; });
	}

	R num;
    memcpy(&num, &active_buffer.data[buffer_i], sizeof(R));
    buffer_i += sizeof(R);

	// out of numbers in current buffer
	if (buffer_i >= buf_limit) {
		active_buffer.ready = false;

		// enqueue read data into the empty buffer, generate future numbers
        queue.enqueueReadBuffer(random_buf, CL_FALSE, 0, buf_size, active_buffer.data.data(), NULL, &active_buffer.ready_event);
		queue.enqueueNDRangeKernel(k_generate, 0, cl::NDRange(nthreads), cl::NullRange);
		active_buffer.ready_event.setCallback(CL_COMPLETE, set_flag, &active_buffer.ready);

		// switch active buffer
		active = !active;
		buffer_i = 0;
	}
    return num;
}

/*
template specialization for floating point numbers
*/

template <>
float RandGPU::rand<float>()
{
    return rand<uint32_t>() / (float) UINT32_MAX;
}

template <>
double RandGPU::rand<double>()
{
    return rand<uint64_t>() / (double) UINT64_MAX;
}

template <>
long double RandGPU::rand<long double>()
{
    return rand<uint64_t>() / (long double) UINT64_MAX;
}

size_t RandGPU::buffer_size()
{
    return buf_size;
}

/*
C function wrappers
*/

vector<RandGPU> randGPUs;

extern "C" {

int rand_gpu_new(uint32_t multi)
{
    init_lock.lock();
    randGPUs.push_back(multi);
    return randGPUs.size() - 1;
    init_lock.unlock();
}

void rand_gpu_delete(int rng)
{
    init_lock.lock();
    randGPUs.erase(randGPUs.begin() + rng);
    init_lock.unlock();
}


uint64_t rand_gpu_u64(int rng) { return randGPUs[rng].rand<uint64_t>(); }
int64_t  rand_gpu_i64(int rng) { return randGPUs[rng].rand<int64_t>();  }
uint32_t rand_gpu_u32(int rng) { return randGPUs[rng].rand<uint32_t>(); }
int32_t  rand_gpu_i32(int rng) { return randGPUs[rng].rand<int32_t>();  }
uint16_t rand_gpu_u16(int rng) { return randGPUs[rng].rand<uint16_t>(); }
int16_t  rand_gpu_i16(int rng) { return randGPUs[rng].rand<int16_t>();  }
uint8_t  rand_gpu_u8(int rng)  { return randGPUs[rng].rand<uint8_t>();  }
int8_t   rand_gpu_i8(int rng)  { return randGPUs[rng].rand<int8_t>();   }


float       rand_gpu_float(int rng)       { return randGPUs[rng].rand<float>();       }
double      rand_gpu_double(int rng)      { return randGPUs[rng].rand<double>();      }
long double rand_gpu_long_double(int rng) { return randGPUs[rng].rand<long double>(); }

size_t rand_gpu_memory() { return mem_all; }

}
