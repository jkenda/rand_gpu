#include "RandGPU.hpp"

#include <thread>
#include <cstdio>
#include <random>
#include <fstream>
#include <sstream>
#include <chrono>
#include <memory>
#include <cstring>
#include <type_traits>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#define __CL_ENABLE_EXCEPTIONS
#include "../include/cl.hpp"

#ifdef _WIN32
#define KERNEL_PATH "kernels\\server.cl"
#else
#define KERNEL_PATH "kernels/server.cl"
#endif

#define MAX_PLATFORMS 10
#define MAX_DEVICES 10
#define PLATFORM 0
#define DEVICE 0

#define FLOAT_MULTI  (1.0f / UINT32_MAX)
#define DOUBLE_MULTI (1.0  / UINT64_MAX)

#define TYCHE_I_STATE_SIZE (4 * sizeof(cl_uint))

using std::vector;
using std::size_t;

static std::mutex buffer_ready_lock;
static std::condition_variable buffer_ready_cond;


RandGPU &RandGPU::instance(size_t multi)
{
    static RandGPU inst(multi);
    return inst;
}

RandGPU::RandGPU(size_t multi)
{
	// get platforms and devices
	vector<cl::Platform> platforms;
	vector<cl::Device> devices;
    cl::Platform::get(&platforms);
    platforms[PLATFORM].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    cl::Device device = devices[DEVICE];

    // get device info
    uint32_t max_cu = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    uint32_t max_wg_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    uint32_t workgroup_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    nthreads = multi * max_cu * max_wg_size;
    buf_size = nthreads * sizeof(cl_ulong);
    buf_limit = buf_size - sizeof(cl_ulong);
    global_range = cl::NDRange(nthreads);
    local_range = cl::NDRange(workgroup_size);

    // resize buffers
    buffer[0].data.resize(buf_size * sizeof(uint8_t));
    buffer[1].data.resize(buf_size * sizeof(uint8_t));

    // create context and command queue
    context = cl::Context({ device });
    queue = cl::CommandQueue(context, device);

    // read kernel file
    std::ifstream fstream = std::ifstream(KERNEL_PATH);
    std::stringstream sstream;
    sstream << fstream.rdbuf();
    std::string src = sstream.str();

    // build program
    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length()));
    cl::Program program = cl::Program(context, sources);
    program.build({ device }, "");

    // create kernels
    k_init = cl::Kernel(program, "init");
    k_generate = cl::Kernel(program, "generate64");

    // create buffers
    state_buf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, nthreads * sizeof(TYCHE_I_STATE_SIZE));
    random_buf = cl::Buffer(context, CL_MEM_WRITE_ONLY, buf_size * sizeof(uint8_t));

	// generate seeds
	vector<cl_ulong> seeds(nthreads);
    std::random_device rd;
	std::mt19937_64 generator(rd());
    for (cl_ulong &seed : seeds) {
        seed = generator();
    }
    cl::Buffer seed_buffer(context, seeds.begin(), seeds.end(), false);

	// initialize RNG
    cl::Event e;
    k_init.setArg(0, state_buf);
    k_init.setArg(0, seed_buffer);
    k_generate.setArg(0, state_buf);
    k_generate.setArg(1, random_buf);
    queue.enqueueNDRangeKernel(k_init, 0, global_range, local_range, NULL, &e);
    e.wait();

    // fill both buffers
    queue.enqueueNDRangeKernel(k_generate, 0, global_range, local_range);
    queue.enqueueReadBuffer(random_buf, CL_TRUE, 0, buf_size * sizeof(uint8_t), buffer[0].data.data());
    queue.enqueueNDRangeKernel(k_generate, 0, global_range, local_range);
    queue.enqueueReadBuffer(random_buf, CL_TRUE, 0, buf_size * sizeof(uint8_t), buffer[1].data.data());
    buffer[0].ready = true;
    buffer[1].ready = true;

	// generate future numbers
    queue.enqueueNDRangeKernel(k_generate, 0, global_range, local_range);
}

void set_ready_flag(cl_event e, cl_int status, void *data)
{
    std::lock_guard<std::mutex> lock(buffer_ready_lock);
	*(std::atomic<bool> *) data = true;
    buffer_ready_cond.notify_one();
}

template <typename R>
R RandGPU::rand()
{
    if (buffer_i == 0) {
		// check if buffer is ready
		std::unique_lock<std::mutex> lock(buffer_ready_lock);
        buffer_ready_cond.wait(lock, [&] { return buffer[active_buffer].ready; });
	}

	R num;
    std::memcpy(&num, &buffer[active_buffer].data[buffer_i], sizeof(R));
    buffer_i += sizeof(R);

	// out of numbers in current buffer
	if (buffer_i >= buf_limit) {
		buffer[active_buffer].ready = false;
		// enqueue read data into the empty buffer, generate future numbers
        queue.enqueueReadBuffer(random_buf, CL_FALSE, 0, buf_size * sizeof(uint8_t), buffer[active_buffer].data.data(), NULL, &buffer_ready_event);
		queue.enqueueNDRangeKernel(k_generate, 0, global_range, local_range);
		buffer_ready_event.setCallback(CL_COMPLETE, set_ready_flag, (void *) &buffer[active_buffer].ready);

		// switch active buffer
		active_buffer = 1^active_buffer;
		buffer_i = 0;
	}
    return num;
}

template <>
float RandGPU::rand<float>()
{
    return FLOAT_MULTI * rand<uint32_t>();
}

template <>
double RandGPU::rand<double>()
{
    return DOUBLE_MULTI * rand<uint64_t>();
}

size_t RandGPU::buffer_size()
{
    return buf_size;
}

/*
C functions
*/

RandGPU *rand_inst = nullptr;

extern "C" {

void rand_gpu_init(uint32_t multi)
{
    rand_inst = &RandGPU::instance(multi);
}


int64_t  rand_gpu_i64() { return rand_inst->rand<int64_t>();  }
int32_t  rand_gpu_i32() { return rand_inst->rand<int32_t>();  }
int16_t  rand_gpu_i16() { return rand_inst->rand<int16_t>();  }
uint64_t rand_gpu_u64() { return rand_inst->rand<uint64_t>(); }
uint32_t rand_gpu_u32() { return rand_inst->rand<uint32_t>(); }
uint16_t rand_gpu_u16() { return rand_inst->rand<uint16_t>(); }

long  rand_gpu_long()  { return rand_gpu_i64(); }
int   rand_gpu_int()   { return rand_gpu_i32(); }
short rand_gpu_short() { return rand_gpu_i16(); }
unsigned long  rand_gpu_ulong()  { return rand_gpu_u64(); }
unsigned int   rand_gpu_uint()   { return rand_gpu_u32(); }
unsigned short rand_gpu_ushort() { return rand_gpu_u16(); }

float  rand_gpu_float()  { return rand_inst->rand<float>(); }
double rand_gpu_double() { return rand_inst->rand<double>(); }

size_t rand_gpu_bufsiz() { return rand_inst->buffer_size(); }

}
