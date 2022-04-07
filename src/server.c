#include "server.h"
#include "util.h"

#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/random.h>

typedef union{
	struct{
		uint a,b,c,d;
	};
	ulong res;
} tyche_i_state;


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

#define MAX_PLATFORMS 10
#define MAX_DEVICES 10
#define PLATFORM 0
#define DEVICE 0

#define BUFFER_SIZE (1024*4)

static const size_t WORK_SIZE = BUFFER_SIZE;
static const size_t WG_SIZE = 256;

#define TYCHE_I_FLOAT_MULTI  5.4210108624275221700372640e-20f
#define TYCHE_I_DOUBLE_MULTI 5.4210108624275221700372640e-20

cl_context __cl_context;
cl_command_queue __cl_queue;
cl_program __cl_program;
cl_kernel __cl_k_init;
cl_kernel __cl_k_generate;
cl_mem __cl_random_buf;
cl_mem __cl_state_buf;

uint64_t buffer[2][BUFFER_SIZE];
pthread_mutex_t buffer_lock[2] = { PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER };

uint32_t active_buffer = 0;
size_t buffer_i = 0;

bool finished = false;
bool fetch = false;
uint32_t fetch_buffer;
pthread_cond_t fetch_cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t fetch_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_t thread_id;

void *__handle_gpu(void *arg)
{
	while (!finished)
	{
		pthread_mutex_lock(&fetch_lock);
		while (!fetch)
			pthread_cond_wait(&fetch_cond, &fetch_lock);
		pthread_mutex_lock(&buffer_lock[fetch_buffer]);
		//printf("\nfetching %u\n", fetch_buffer); fflush(stdout);

		clEnqueueReadBuffer(__cl_queue, __cl_random_buf, CL_FALSE, 0, sizeof(buffer[1]), buffer[1], 0, NULL, NULL);

		fetch = false;
		pthread_mutex_unlock(&buffer_lock[fetch_buffer]);
		pthread_mutex_unlock(&fetch_lock);
		clEnqueueNDRangeKernel(__cl_queue, __cl_k_generate, 1, 0, &WORK_SIZE, &WG_SIZE, 0, NULL, NULL);
	}
}

int __gpu_init()
{
	cl_int ret;
	cl_int status = 0;

	// read kernel source file
	FILE *fp = fopen("kernels/server.cl", "r");
	fseek(fp, 0, SEEK_END);
	long fsize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	char *src = malloc(fsize+1);
	fread(src, fsize, 1, fp);
	src[fsize] = '\0';
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
    ret = clBuildProgram(__cl_program, num_devices, device_id, COMPILE_OPTS, NULL, NULL); status += ret;
	if (ret != 0) {
		print_cl_err(__cl_program, device_id[0]);
		exit(status);
	}

    // create kernels
    __cl_k_init     = clCreateKernel(__cl_program, "init", &ret); status += ret;
    __cl_k_generate = clCreateKernel(__cl_program, "generate", &ret); status += ret;

	// create buffer
	__cl_random_buf = clCreateBuffer(__cl_context, CL_MEM_WRITE_ONLY, BUFFER_SIZE * sizeof(cl_ulong), NULL, &ret);
	__cl_state_buf  = clCreateBuffer(__cl_context, CL_MEM_READ_WRITE, BUFFER_SIZE * sizeof(tyche_i_state), NULL, &ret);

	// start thread
	pthread_create(&thread_id, NULL, __handle_gpu, NULL);

	return status;
}

int rand_init()
{
	// initialize GPU
	int ret = __gpu_init();
	
	// generate seeds
	cl_ulong seed[BUFFER_SIZE];
	getrandom(seed, sizeof(seed), 0);
	cl_mem seed_buffer = clCreateBuffer(__cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
		sizeof(seed), seed, &ret);

	// initialize RNG
	clSetKernelArg(__cl_k_init, 0, sizeof(cl_mem), &__cl_state_buf);
	clSetKernelArg(__cl_k_init, 1, sizeof(cl_mem), &seed_buffer);
	clSetKernelArg(__cl_k_generate, 0, sizeof(cl_mem), &__cl_state_buf);
	clSetKernelArg(__cl_k_generate, 1, sizeof(cl_mem), &__cl_random_buf);
	clEnqueueNDRangeKernel(__cl_queue, __cl_k_init, 1, 0, &WORK_SIZE, &WG_SIZE, 0, NULL, NULL);

	// fill both buffers

	clEnqueueNDRangeKernel(__cl_queue, __cl_k_generate, 1, 0, &WORK_SIZE, &WG_SIZE, 0, NULL, NULL);
	clEnqueueReadBuffer(__cl_queue, __cl_random_buf, CL_TRUE, 0, sizeof(buffer[0]), buffer[0], 0, NULL, NULL);

	clEnqueueNDRangeKernel(__cl_queue, __cl_k_generate, 1, 0, &WORK_SIZE, &WG_SIZE, 0, NULL, NULL);
	clEnqueueReadBuffer(__cl_queue, __cl_random_buf, CL_TRUE, 0, sizeof(buffer[1]), buffer[1], 0, NULL, NULL);

	clEnqueueNDRangeKernel(__cl_queue, __cl_k_generate, 1, 0, &WORK_SIZE, &WG_SIZE, 0, NULL, NULL);

	// result: local buffers and GPU buffer are all filled
	clReleaseMemObject(seed_buffer);

	return ret;
}

int rand_clean()
{
	finished = true;
	pthread_kill(thread_id, 0);

    clFlush(__cl_queue);
	clFinish(__cl_queue);
	clReleaseMemObject(__cl_random_buf);
	clReleaseMemObject(__cl_state_buf);
	clReleaseKernel(__cl_k_init);
	clReleaseKernel(__cl_k_generate);
	clReleaseProgram(__cl_program);
	clReleaseCommandQueue(__cl_queue);
	clReleaseContext(__cl_context);
}

cl_ulong _rand_get()
{
	pthread_mutex_lock(&buffer_lock[active_buffer]);
	cl_ulong num = buffer[active_buffer][buffer_i++];
	if (buffer_i == BUFFER_SIZE) {
		pthread_mutex_lock(&fetch_lock);
		fetch = true;
		fetch_buffer = active_buffer;
		active_buffer = 1^active_buffer;
		buffer_i = 0;
		pthread_cond_signal(&fetch_cond);
		//printf("\nactive buffer: %u\n", active_buffer); fflush(stdout);
		pthread_mutex_unlock(&fetch_lock);
		pthread_mutex_unlock(&buffer_lock[fetch_buffer]);
	}
	else {
		pthread_mutex_unlock(&buffer_lock[active_buffer]);
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

float rand_get_float() { return TYCHE_I_FLOAT_MULTI * _rand_get(); }

double rand_get_double() { return TYCHE_I_DOUBLE_MULTI * _rand_get(); }
