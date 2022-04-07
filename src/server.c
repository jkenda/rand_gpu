#include "server.h"
#include "util.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // cl.hpp
#define CL_TARGET_OPENCL_VERSION 210
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.h>
#endif

#ifdef _WIN32
#define GENERATOR_LOCATION "RandomCL\\generators\\"
#else
#define GENERATOR_LOCATION "RandomCL/generators/"
#endif

#define COMPILE_OPTS "-I " GENERATOR_LOCATION

#define MAX_SRC_SIZE (16384)

#define MAX_PLATFORMS 10
#define MAX_DEVICES 10
#define PLATFORM 0
#define DEVICE 0

#define BUFFER_SIZE 1024

static const size_t WORK_SIZE = BUFFER_SIZE;
static const size_t WG_SIZE = 256;

#define TYCHE_I_FLOAT_MULTI 5.4210108624275221700372640e-20f
#define TYCHE_I_DOUBLE_MULTI 5.4210108624275221700372640e-20

cl_context __cl_context;
cl_command_queue __cl_queue;
cl_program __cl_program;
cl_kernel __cl_k_init;
cl_kernel __cl_k_generate;
cl_kernel __cl_k_get;
cl_mem __buffer;

uint64_t buffers[2][BUFFER_SIZE];
pthread_mutex_t buffer_locks[2] = { PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER };

uint32_t active_buffer = 0;
size_t buffer_i = 0;

bool finished = false;
bool num_ready = false;
bool fetch_buffer = false;
pthread_cond_t fetch_buffer_cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t fetch_buffer_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_t thread_id;

void *__handle_gpu(void *arg)
{
	while (!finished)
	{
		pthread_cond_wait(&fetch_buffer_cond, &fetch_buffer_lock);
		uint32_t inactive_buffer = active_buffer ^ 1;
		pthread_mutex_lock(&buffer_locks[inactive_buffer]);

		clEnqueueNDRangeKernel(__cl_queue, __cl_k_get, 1, 0, &WORK_SIZE, &WG_SIZE, 0, NULL, NULL);
		clEnqueueReadBuffer(__cl_queue, __buffer, CL_TRUE, 0, sizeof(buffers[1]), buffers[1], 0, NULL, NULL);

		pthread_mutex_unlock(&buffer_locks[inactive_buffer]);
		pthread_mutex_unlock(&fetch_buffer_lock);
		clEnqueueNDRangeKernel(__cl_queue, __cl_k_generate, 1, 0, &WORK_SIZE, &WG_SIZE, 0, NULL, NULL);
	}
}

int __gpu_init()
{
	cl_int ret;
	cl_int status = 0;

	// read kernel source file
	char *src = malloc(MAX_SRC_SIZE);
	FILE *fp = fopen("kernels/server.cl", "r");
	fgets(src, MAX_SRC_SIZE, fp);
	fclose(fp);

	// get platforms
	cl_platform_id	platform_id[MAX_PLATFORMS];
	cl_uint			num_platforms;
	status += clGetPlatformIDs(MAX_PLATFORMS, platform_id, &num_platforms);

	// get devices
    cl_device_id	device_id[MAX_DEVICES];
	cl_uint			num_devices;
    status += clGetDeviceIDs(platform_id[PLATFORM], CL_DEVICE_TYPE_GPU, MAX_DEVICES, device_id, &num_devices);

	// crate context, command queue, program
   	__cl_context = clCreateContext(NULL, num_devices, device_id, NULL, NULL, &ret); status += ret;
   	__cl_queue   = clCreateCommandQueue(__cl_context, device_id[DEVICE], 0, &ret); status += ret;
    __cl_program = clCreateProgramWithSource(__cl_context, 1, (const char **) &src, NULL, &ret); status += ret;
	free(src);

    // build program
    status += clBuildProgram(__cl_program, num_devices, device_id, COMPILE_OPTS, NULL, NULL);
	if (status != 0) {
		print_cl_err(__cl_program, device_id[0]);
		exit(status);
	}

    // create kernels
    __cl_k_init     = clCreateKernel(__cl_program, "init", &ret); status += ret;
    __cl_k_generate = clCreateKernel(__cl_program, "generate", &ret); status += ret;
    __cl_k_get      = clCreateKernel(__cl_program, "get", &ret); status += ret;

	// create buffer
	__buffer = clCreateBuffer(__cl_context, CL_MEM_WRITE_ONLY, BUFFER_SIZE * sizeof(cl_ulong), NULL, &ret);
	num_ready = true;

	return status;
}

int rand_init()
{
	// initialize GPU
	int ret = __gpu_init();
	
	// generate seeds
	srand((unsigned int) time(NULL));
	cl_ulong seed[BUFFER_SIZE];
	for (int i = 0; i < BUFFER_SIZE; i++) {
		seed[i] = rand();
	}
	cl_mem seed_buffer = clCreateBuffer(__cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
		sizeof(seed), seed, &ret);

	// initialize RNG
	clSetKernelArg(__cl_k_init, 0, sizeof(cl_mem), &seed_buffer);
	clEnqueueNDRangeKernel(__cl_queue, __cl_k_init, 1, 0, &WORK_SIZE, &WG_SIZE, 0, NULL, NULL);

	// fill both buffers
	clSetKernelArg(__cl_k_get, 0, sizeof(cl_mem), &__buffer);

	clEnqueueNDRangeKernel(__cl_queue, __cl_k_generate, 1, 0, &WORK_SIZE, &WG_SIZE, 0, NULL, NULL);
	clEnqueueNDRangeKernel(__cl_queue, __cl_k_get, 1, 0, &WORK_SIZE, &WG_SIZE, 0, NULL, NULL);
	clEnqueueReadBuffer(__cl_queue, __buffer, CL_TRUE, 0, sizeof(buffers[0]), buffers[0], 0, NULL, NULL);

	clEnqueueNDRangeKernel(__cl_queue, __cl_k_generate, 1, 0, &WORK_SIZE, &WG_SIZE, 0, NULL, NULL);
	clEnqueueNDRangeKernel(__cl_queue, __cl_k_get, 1, 0, &WORK_SIZE, &WG_SIZE, 0, NULL, NULL);
	clEnqueueReadBuffer(__cl_queue, __buffer, CL_TRUE, 0, sizeof(buffers[1]), buffers[1], 0, NULL, NULL);

	clEnqueueNDRangeKernel(__cl_queue, __cl_k_generate, 1, 0, &WORK_SIZE, &WG_SIZE, 0, NULL, NULL);

	// result: local buffers and GPU buffer are all filled

	return ret;
}

int rand_clean()
{
	finished = true;
	pthread_join(thread_id, NULL);

    clFlush(__cl_queue);
	clFinish(__cl_queue);
	clReleaseKernel(__cl_k_init);
	clReleaseKernel(__cl_k_generate);
	clReleaseKernel(__cl_k_get);
	clReleaseProgram(__cl_program);
	clReleaseCommandQueue(__cl_queue);
	clReleaseContext(__cl_context);
}

void _get_buffer()
{

}

cl_ulong _rand_get()
{
	cl_ulong num = buffers[active_buffer][buffer_i++];
	if (buffer_i == BUFFER_SIZE) {
		active_buffer ^= 1;
		pthread_mutex_lock(&fetch_buffer_lock);
		pthread_cond_signal(&fetch_buffer_cond);
	}
	return num;
}

uint64_t rand_get_u64() { return _rand_get(); }

int64_t rand_get_i64() { return (int64_t) _rand_get(); }

uint32_t rand_get_u32() { return (uint32_t) _rand_get(); }

int32_t rand_get_i32() { return (int32_t) _rand_get(); }

uint16_t rand_get_u16() { return (uint16_t) _rand_get(); }

int16_t rand_get_i16() { return (int16_t) _rand_get(); }


long rand_get_long() { return rand_get_i64(); }

unsigned long rand_get_unsigned_long() { return rand_get_u64(); }

int rand_get_int() { return rand_get_i32(); }

unsigned int rand_get_uint() { return rand_get_u32(); }

short rand_get_short() { return rand_get_i16(); }

unsigned short rand_get_ushort() { return rand_get_u16(); }

float rand_get_float() { return (float) _rand_get() * TYCHE_I_FLOAT_MULTI; }

double rand_get_double() { return (double) _rand_get() * TYCHE_I_DOUBLE_MULTI; }
