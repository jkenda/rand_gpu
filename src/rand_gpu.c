#include "rand_gpu.h"
#include "util.h"

#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/random.h>
#include <assert.h>

typedef union{
	struct{
		uint a,b,c,d;
	};
	ulong res;
} tyche_i_state;


#define CL_TARGET_OPENCL_VERSION 210
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.h>
#endif

#ifdef _WIN32
#define GENERATOR_LOCATION "RandomCL\\generators\\"
#define KERNEL_PATH "kernels\\server.cl"
#else
#define GENERATOR_LOCATION "RandomCL/generators/"
#define KERNEL_PATH "kernels/server.cl"
#endif

#define COMPILE_OPTS "-I " GENERATOR_LOCATION

#define MAX_PLATFORMS 10
#define MAX_DEVICES 10
#define PLATFORM 0
#define DEVICE 0

#define TYCHE_I_FLOAT_MULTI  5.4210108624275221700372640e-20f
#define TYCHE_I_DOUBLE_MULTI 5.4210108624275221700372640e-20

size_t WG_SIZE = -1;
size_t BUFFER_SIZE = -1;

cl_context __cl_context;
cl_command_queue __cl_queue;
cl_program __cl_program;
cl_kernel __cl_k_init;
cl_kernel __cl_k_generate;
cl_mem __cl_random_buf;
cl_mem __cl_state_buf;

cl_ulong *buffer[2];
uint_fast32_t active_buffer = 0;
uint_fast32_t buffer_i = 0;

int __gpu_init()
{
	cl_int ret;
	cl_int status = 0;

	// read kernel source file
	FILE *fp = fopen(KERNEL_PATH, "r");
	if (fp == NULL) { fprintf(stderr, "Could not find file %s", KERNEL_PATH); exit(1); }

	fseek(fp, 0, SEEK_END);
	long fsize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	char *src = malloc(fsize+1);
	status += fread(src, fsize, 1, fp);
	src[fsize] = '\0';
	fclose(fp);

	// get platforms
	cl_platform_id	platform_id[MAX_PLATFORMS];
	cl_uint			num_platforms;
	status += clGetPlatformIDs(MAX_PLATFORMS, platform_id, &num_platforms);
	if (num_platforms == 0) { fputs("Could not find any OpenCL platforms.\n", stderr); exit(2); }

	// get devices
    cl_device_id	device_id[MAX_DEVICES];
	cl_uint			num_devices;
    status += clGetDeviceIDs(platform_id[PLATFORM], CL_DEVICE_TYPE_GPU, MAX_DEVICES, device_id, &num_devices);
	if (num_devices == 0) { fputs("Could not find any GPUs.\n", stderr); exit(3); }

	// get GPU info
	gpu_info_t info = gpu_info(device_id[DEVICE]);

	WG_SIZE = info.max_work_group_sizes;
	BUFFER_SIZE = info.max_work_group_sizes * info.compute_units;

	buffer[0] = malloc(BUFFER_SIZE * sizeof(cl_ulong));
	buffer[1] = malloc(BUFFER_SIZE * sizeof(cl_ulong));

	// crate context, command queue, program
   	__cl_context = clCreateContext(NULL, num_devices, device_id, NULL, NULL, &ret); status += ret;
   	__cl_queue   = clCreateCommandQueue(__cl_context, device_id[DEVICE], 0, &ret); status += ret;
    __cl_program = clCreateProgramWithSource(__cl_context, 1, (const char **) &src, NULL, &ret); status += ret;
	free(src);

    // build program
    ret = clBuildProgram(__cl_program, num_devices, device_id, COMPILE_OPTS, NULL, NULL); status += ret;
	if (ret != 0) {
		print_cl_err(__cl_program, device_id[DEVICE]);
		exit(status);
	}

    // create kernels
    __cl_k_init     = clCreateKernel(__cl_program, "init", &ret); status += ret;
    __cl_k_generate = clCreateKernel(__cl_program, "generate", &ret); status += ret;

	// create buffer
	__cl_random_buf = clCreateBuffer(__cl_context, CL_MEM_WRITE_ONLY, BUFFER_SIZE * sizeof(cl_ulong), NULL, &ret);
	__cl_state_buf  = clCreateBuffer(__cl_context, CL_MEM_READ_WRITE, BUFFER_SIZE * sizeof(tyche_i_state), NULL, &ret);

	return status;
}

// TODO: rand_init_32 (transfer 32 bit numbers instead of 64 bit ones)

int rand_init()
{
	// initialize GPU
	int ret;
	int status = __gpu_init();

	// generate seeds
	cl_ulong *seed = malloc(BUFFER_SIZE * sizeof(cl_ulong));
	status += getrandom(seed, sizeof(seed), 0);
	cl_mem seed_buffer = clCreateBuffer(__cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
		 BUFFER_SIZE * sizeof(cl_ulong), seed, &ret); status += ret;
	free(seed);

	// initialize RNG
	status += clSetKernelArg(__cl_k_init, 0, sizeof(cl_mem), &__cl_state_buf);
	status += clSetKernelArg(__cl_k_init, 1, sizeof(cl_mem), &seed_buffer);
	status += clSetKernelArg(__cl_k_generate, 0, sizeof(cl_mem), &__cl_state_buf);
	status += clSetKernelArg(__cl_k_generate, 1, sizeof(cl_mem), &__cl_random_buf);
	status += clEnqueueNDRangeKernel(__cl_queue, __cl_k_init, 1, 0, &BUFFER_SIZE, &WG_SIZE, 0, NULL, NULL);
	status += clFinish(__cl_queue);
	status += clReleaseMemObject(seed_buffer);

	// fill both buffers

	active_buffer = 1;
	status += clEnqueueNDRangeKernel(__cl_queue, __cl_k_generate, 1, 0, &BUFFER_SIZE, &WG_SIZE, 0, NULL, NULL);
	status += clEnqueueReadBuffer(__cl_queue, __cl_random_buf, CL_TRUE, 0, BUFFER_SIZE * sizeof(cl_ulong), buffer[0], 0, NULL, NULL);

	active_buffer = 0;
	status += clEnqueueNDRangeKernel(__cl_queue, __cl_k_generate, 1, 0, &BUFFER_SIZE, &WG_SIZE, 0, NULL, NULL);
	status += clEnqueueReadBuffer(__cl_queue, __cl_random_buf, CL_TRUE, 0, BUFFER_SIZE * sizeof(cl_ulong), buffer[1], 0, NULL, NULL);

	status += clEnqueueNDRangeKernel(__cl_queue, __cl_k_generate, 1, 0, &BUFFER_SIZE, &WG_SIZE, 0, NULL, NULL);

	return status;
}

int rand_clean()
{
	int status = 0;
    status += clFlush(__cl_queue);
	status += clFinish(__cl_queue);
	status += clReleaseMemObject(__cl_random_buf);
	status += clReleaseMemObject(__cl_state_buf);
	status += clReleaseKernel(__cl_k_init);
	status += clReleaseKernel(__cl_k_generate);
	status += clReleaseCommandQueue(__cl_queue);
	status += clReleaseProgram(__cl_program);
	status += clReleaseContext(__cl_context);
	free(buffer[0]);
	free(buffer[1]);
	return status;
}

size_t __rand_gpu_bufsiz() { return BUFFER_SIZE; }

/**
 * @brief Retrieves next random number,
 *        switches buffers if necessary
 * 
 * @return cl_ulong 
 */
cl_ulong __rand_gpu()
{
	cl_ulong num = buffer[active_buffer][buffer_i++];

	// out of numbers in current buffer
	if (buffer_i == BUFFER_SIZE) {
		// switch active buffer
		active_buffer = 1^active_buffer;
		buffer_i = 0;

		// read data into inactive buffer, generate future numbers
		clEnqueueReadBuffer(__cl_queue, __cl_random_buf, CL_FALSE, 0, 
			BUFFER_SIZE * sizeof(cl_ulong), buffer[1^active_buffer], 0, NULL, NULL);
		clEnqueueNDRangeKernel(__cl_queue, __cl_k_generate, 1, 0, &BUFFER_SIZE, &WG_SIZE, 0, NULL, NULL);
	}
	return num;
}

uint64_t rand_gpu_u64() { return __rand_gpu(); }

int64_t rand_gpu_i64() { return (int64_t) __rand_gpu(); }

uint32_t rand_gpu_u32() { return (uint32_t) __rand_gpu(); }

int32_t rand_gpu_i32() { return (int32_t) __rand_gpu(); }

uint16_t rand_gpu_u16() { return (uint16_t) __rand_gpu(); }

int16_t rand_gpu_i16() { return (int16_t) __rand_gpu(); }


long rand_gpu_long() { return rand_gpu_i64(); }

unsigned long rand_gpu_unsigned_long() { return rand_gpu_u64(); }

int rand_gpu_int() { return rand_gpu_i32(); }

unsigned int rand_gpu_uint() { return rand_gpu_u32(); }

short rand_gpu_short() { return rand_gpu_i16(); }

unsigned short rand_gpu_ushort() { return rand_gpu_u16(); }

float rand_gpu_float() { return TYCHE_I_FLOAT_MULTI * __rand_gpu(); }

double rand_gpu_double() { return TYCHE_I_DOUBLE_MULTI * __rand_gpu(); }
