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
#include <memory>
#include <cstring>
#include "exceptions.hpp"
#include "../kernel.hpp"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#include "../include/cl.hpp"

#define TYCHE_I_STATE_SIZE (4 * sizeof(cl_uint))

using namespace std;

static mutex buffer_ready_lock;
static condition_variable buffer_ready_cond;

#define TYCHE_I_FLOAT_MULTI 5.4210108624275221700372640e-20f
#define TYCHE_I_DOUBLE_MULTI 5.4210108624275221700372640e-20

/*
FIXME: segfault on exit (double free)
FIXME: numbers not random enough
*/


RandGPU &RandGPU::instance(size_t multi)
{
    static RandGPU inst(multi);
    return inst;
}

RandGPU::RandGPU(size_t multi)
:
    active(0), buffer_i(sizeof(cl_ulong))
{
	cl::Platform platform;
	vector<cl::Device> devices;

	// get platforms
    int status = cl::Platform::get(&platform);
    if (status != CL_SUCCESS) throw RandGPUException("No OpenCL platforms found.");
    
    // get devices
    status = platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (status != CL_SUCCESS) throw RandGPUException("No OpenCL devices found.");

    // create devices
    cl::Device device0 = devices.at(0);
    cl::Device device1 = devices.size() > 1 ? devices.at(1) : device0;

    // get device info
    uint32_t max_cu = device0.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    uint32_t max_wg_size = device0.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    wg_size = device0.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    nthreads = multi * max_cu * max_wg_size;
    buf_size = nthreads * sizeof(cl_ulong);
    buf_limit = buf_size - sizeof(cl_ulong);

    // resize buffers
    buffer[0].data.resize(buf_size);
    buffer[1].data.resize(buf_size);

    // create context and command queue
    context = cl::Context(devices);
    buffer[0].queue = cl::CommandQueue(context, device0);
    buffer[1].queue = devices.size() > 1 ? cl::CommandQueue(context, device1) : buffer[0].queue;

    // build program
    cl::Program::Sources sources(1, make_pair(KERNEL_SOURCE.c_str(), KERNEL_SOURCE.length()));
    cl::Program program = cl::Program(context, sources);
    program.build(devices, "");
    status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device0);
    if (status == CL_BUILD_ERROR) {
        std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device0);
        fputs(buildlog.c_str(), stderr);
        throw RandGPUException("Build failed.");
    }

    // create kernels
    k_init = cl::Kernel(program, "init", &status);
    if (status != CL_SUCCESS) throw RandGPUException("Could not create kernel 'init'.");
    k_generate = cl::Kernel(program, "generate", &status);
    if (status != CL_SUCCESS) throw RandGPUException("Could not create kernel 'generate'.");

    // create buffers
    state_buf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, nthreads * TYCHE_I_STATE_SIZE, NULL, &status);
    if (status != CL_SUCCESS) throw RandGPUException("Could not create state buffer.");
    random_buf = cl::Buffer(context, CL_MEM_WRITE_ONLY, buf_size, NULL, &status);
    if (status != CL_SUCCESS) throw RandGPUException("Could not create random buffer.");

	// generate seeds
	vector<cl_ulong> seeds(nthreads);
    random_device rd;
	mt19937_64 generator(rd());
    for (cl_ulong &seed : seeds) {
        seed = generator();
    }
    cl::Buffer seed_buffer(context, seeds.begin(), seeds.end(), true, NULL, &status);
    if (status != CL_SUCCESS) throw RandGPUException("Could not create seed buffer.");

	// initialize RNG
    cl::Event e;
    k_init.setArg(0, state_buf);
    k_init.setArg(1, seed_buffer);
    k_generate.setArg(0, state_buf);
    k_generate.setArg(1, random_buf);
    status = buffer[0].queue.enqueueNDRangeKernel(k_init, 0, cl::NDRange(nthreads), cl::NullRange, NULL, &e);
    if (status != CL_SUCCESS) throw RandGPUException("Could not start init kernel", status);
    e.wait();
    status = buffer[1].queue.enqueueNDRangeKernel(k_init, 0, cl::NDRange(nthreads), cl::NullRange, NULL, &e);
    if (status != CL_SUCCESS) throw RandGPUException("Could not start init kernel", status);
    e.wait();

    // fill both buffers
    status = buffer[0].queue.enqueueNDRangeKernel(k_generate, 0, cl::NDRange(nthreads), cl::NullRange, NULL, &e);
    if (status != CL_SUCCESS) throw RandGPUException("Could not start generate kernel", status);
    e.wait();
    status = buffer[0].queue.enqueueReadBuffer(random_buf, CL_TRUE, 0, buf_size, buffer[0].data.data());
    if (status != CL_SUCCESS) throw RandGPUException("Could not start read buffer", status);
    status = buffer[1].queue.enqueueNDRangeKernel(k_generate, 0, cl::NDRange(nthreads), cl::NullRange, NULL, &e);
    if (status != CL_SUCCESS) throw RandGPUException("Could not start generate kernel", status);
    e.wait();
    status = buffer[1].queue.enqueueReadBuffer(random_buf, CL_TRUE, 0, buf_size, buffer[1].data.data());
    if (status != CL_SUCCESS) throw RandGPUException("Could not start read buffer", status);
    buffer[0].ready = true;
    buffer[1].ready = true;

	// generate future numbers
    buffer[0].queue.enqueueNDRangeKernel(k_generate, 0, cl::NDRange(nthreads), cl::NullRange);
    buffer[1].queue.enqueueNDRangeKernel(k_generate, 0, cl::NDRange(nthreads), cl::NullRange);
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
        active_buffer.queue.enqueueReadBuffer(random_buf, CL_FALSE, 0, buf_size, active_buffer.data.data(), NULL, &buffer[active].ready_event);
		active_buffer.queue.enqueueNDRangeKernel(k_generate, 0, cl::NDRange(nthreads), cl::NullRange);
		active_buffer.ready_event.setCallback(CL_COMPLETE, set_flag, &active_buffer.ready);

		// switch active buffer
		active = !active;
		buffer_i = 0;
	}
    return num;
}

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

size_t RandGPU::buffer_size()
{
    return buf_size;
}

/*
C function wrappers
*/

unique_ptr<RandGPU> rand_inst;

extern "C" {

void rand_gpu_init(uint32_t multi)
{
    rand_inst = unique_ptr<RandGPU>(&RandGPU::instance(multi));
}


int64_t  rand_gpu_i64() { return rand_inst->rand<int64_t>();  }
int32_t  rand_gpu_i32() { return rand_inst->rand<int32_t>();  }
int16_t  rand_gpu_i16() { return rand_inst->rand<int16_t>();  }
int8_t   rand_gpu_i8()  { return rand_inst->rand<int8_t>();   }
uint64_t rand_gpu_u64() { return rand_inst->rand<uint64_t>(); }
uint32_t rand_gpu_u32() { return rand_inst->rand<uint32_t>(); }
uint16_t rand_gpu_u16() { return rand_inst->rand<uint16_t>(); }
uint8_t  rand_gpu_u8()  { return rand_inst->rand<uint8_t>();  }

long  rand_gpu_long()  { return rand_gpu_i64(); }
int   rand_gpu_int()   { return rand_gpu_i32(); }
short rand_gpu_short() { return rand_gpu_i16(); }
char  rand_gpu_char()  { return rand_gpu_i8();  }
unsigned long  rand_gpu_ulong()  { return rand_gpu_u64(); }
unsigned int   rand_gpu_uint()   { return rand_gpu_u32(); }
unsigned short rand_gpu_ushort() { return rand_gpu_u16(); }
unsigned char  rand_gpu_uchar()  { return rand_gpu_u8();  }

float  rand_gpu_float()  { return rand_inst->rand<float>(); }
double rand_gpu_double() { return rand_inst->rand<double>(); }

size_t rand_gpu_bufsiz() { return rand_inst->buffer_size(); }

}
