#include "rand_gpu.h"
#include "util.h"

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/random.h>
#include <assert.h>

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

#define MAX_PLATFORMS 10
#define MAX_DEVICES 10
#define PLATFORM 0
#define DEVICE 0

#define FLOAT_MULTI_32  (1.0f / UINT32_MAX)
#define FLOAT_MULTI_64  (1.0f / UINT64_MAX)
#define DOUBLE_MULTI_64 (1.0  / UINT64_MAX)

#define TYCHE_I_STATE_SIZE (4 * sizeof(cl_uint))

typedef struct
{
	cl_uint *data;
	cl_bool ready;
}
buffer32_t;

typedef struct
{
	cl_ulong *data;
	cl_bool ready;
}
buffer64_t;


size_t _workgroup_size = -1;
size_t _buffer_size = -1;

cl_context       _cl_context;
cl_command_queue _cl_queue;
cl_program       _cl_program;
cl_kernel        _cl_k_init;
cl_kernel        _cl_k_generate;
cl_mem           _cl_state_buf;
cl_mem           _cl_random_buf;

buffer32_t _buffer32[2] = {0};
buffer64_t _buffer64[2] = {0};
uint_fast32_t _active_buffer32 = 0;
uint_fast32_t _active_buffer64 = 0;
uint_fast32_t _buffer32_i = 0;
uint_fast32_t _buffer64_i = 0;

cl_event _buffer_ready_event;
pthread_mutex_t _buffer_ready_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t _buffer_ready_cond;

/*
	PRIVATE
*/

/* 
FIXME: numbers in the same buffer stay the same
*/

void CL_CALLBACK __set_ready_flag(cl_event e, cl_int status, void *data)
{
	pthread_mutex_lock(&_buffer_ready_lock);
	*(cl_bool *) data = CL_TRUE;
	pthread_cond_signal(&_buffer_ready_cond);
	pthread_mutex_unlock(&_buffer_ready_lock);
}

int __gpu_init(uint32_t which, uint32_t multi)
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

	_workgroup_size = info.max_work_group_sizes;
	_buffer_size = multi * info.max_work_group_sizes * info.compute_units;

	if (which == 64) {
		_buffer64[0].data = malloc(_buffer_size * sizeof(cl_ulong));
		if (_buffer64[0].data == NULL) { printf("Could not allocate %lu bytes of memory\n", _buffer_size * sizeof(cl_ulong)); exit(5); }
		_buffer64[1].data = malloc(_buffer_size * sizeof(cl_ulong));
		if (_buffer64[0].data == NULL) { printf("Could not allocate %lu bytes of memory\n", _buffer_size * sizeof(cl_ulong)); exit(5); }
	}
	else {
		_buffer32[0].data = malloc(_buffer_size * sizeof(cl_uint));
		if (_buffer32[0].data == NULL) { printf("Could not allocate %lu bytes of memory\n", _buffer_size * sizeof(cl_uint)); exit(5); }
		_buffer32[1].data = malloc(_buffer_size * sizeof(cl_uint));
		if (_buffer32[0].data == NULL) { printf("Could not allocate %lu bytes of memory\n", _buffer_size * sizeof(cl_uint)); exit(5); }
	}

	// crate context, command queue, program
   	_cl_context = clCreateContext(NULL, num_devices, device_id, NULL, NULL, &ret);
   	_cl_queue   = clCreateCommandQueue(_cl_context, device_id[DEVICE], 0, &ret);
    _cl_program = clCreateProgramWithSource(_cl_context, 1, (const char **) &src, NULL, &ret);
	free(src);

    // build program
    ret = clBuildProgram(_cl_program, num_devices, device_id, 0, NULL, NULL);
	if (ret != 0) {
		print_cl_err(_cl_program, device_id[DEVICE]);
		exit(status);
	}

    // create kernels
    _cl_k_init     = clCreateKernel(_cl_program, "init", &ret);
	_cl_k_generate = clCreateKernel(_cl_program, (which == 64) ? "generate64" : "generate32", &ret);

	// generate seeds
	cl_ulong seed[_buffer_size];
	status = getrandom(seed, sizeof(seed), 0);
	if (status == -1) { perror("Could not get rangom seeds"); exit(4); }
	cl_mem seed_buffer = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
			_buffer_size * sizeof(cl_ulong), seed, &ret);

	// initialize RNG
	clSetKernelArg(_cl_k_init, 0, sizeof(cl_mem), &_cl_state_buf);
	clSetKernelArg(_cl_k_init, 1, sizeof(cl_mem), &seed_buffer);
	clSetKernelArg(_cl_k_generate, 0, sizeof(cl_mem), &_cl_state_buf);
	clSetKernelArg(_cl_k_generate, 1, sizeof(cl_mem), &_cl_random_buf);
	clEnqueueNDRangeKernel(_cl_queue, _cl_k_init, 1, 0, &_buffer_size, &_workgroup_size, 0, NULL, NULL);
	clFinish(_cl_queue);
	clReleaseMemObject(seed_buffer);

	// create buffers
	_cl_state_buf  = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, _buffer_size * TYCHE_I_STATE_SIZE, NULL, &ret);
	if (which == 64)
		_cl_random_buf = clCreateBuffer(_cl_context, CL_MEM_WRITE_ONLY, _buffer_size * sizeof(cl_ulong), NULL, &ret);
	else
		_cl_random_buf = clCreateBuffer(_cl_context, CL_MEM_WRITE_ONLY, _buffer_size * sizeof(cl_uint), NULL, &ret);

	// fill both buffers
	clEnqueueNDRangeKernel(_cl_queue, _cl_k_generate, 1, 0, &_buffer_size, &_workgroup_size, 0, NULL, NULL);
	if (which == 64) {
		clEnqueueReadBuffer(_cl_queue, _cl_random_buf, CL_TRUE, 0, _buffer_size * sizeof(cl_ulong), _buffer64[0].data, 0, NULL, NULL);
		_buffer64[0].ready = CL_TRUE;
	}
	else {
		clEnqueueReadBuffer(_cl_queue, _cl_random_buf, CL_TRUE, 0, _buffer_size * sizeof(cl_uint), _buffer32[0].data, 0, NULL, NULL);
		_buffer32[0].ready = CL_TRUE;
	}

	clEnqueueNDRangeKernel(_cl_queue, _cl_k_generate, 1, 0, &_buffer_size, &_workgroup_size, 0, NULL, NULL);
	if (which == 64) {
		clEnqueueReadBuffer(_cl_queue, _cl_random_buf, CL_TRUE, 0, _buffer_size * sizeof(cl_ulong), _buffer64[1].data, 0, NULL, NULL);
		_buffer64[1].ready = CL_TRUE;
	}
	else {
		clEnqueueReadBuffer(_cl_queue, _cl_random_buf, CL_TRUE, 0, _buffer_size * sizeof(cl_uint), _buffer32[1].data, 0, NULL, NULL);
		_buffer32[1].ready = CL_TRUE;
	}

	// generate future numbers
	clEnqueueNDRangeKernel(_cl_queue, _cl_k_generate, 1, 0, &_buffer_size, &_workgroup_size, 0, NULL, NULL);

	return status;
}

cl_uint __rand_gpu32()
{
	if (_buffer32_i == 0) {
		// check if buffer is ready
		pthread_mutex_lock(&_buffer_ready_lock);
		while (!_buffer32[_active_buffer32].ready)
			pthread_cond_wait(&_buffer_ready_cond, &_buffer_ready_lock);
		pthread_mutex_unlock(&_buffer_ready_lock);
	}

	cl_uint num = _buffer32[_active_buffer32].data[_buffer32_i++];

	// out of numbers in current buffer
	if (_buffer32_i == _buffer_size) {
		_buffer32[_active_buffer32].ready = CL_FALSE;
		// enqueue read data into the empty buffer, generate future numbers
		clEnqueueReadBuffer(_cl_queue, _cl_random_buf, CL_FALSE, 0, 
			_buffer_size * sizeof(cl_uint), _buffer32[_active_buffer32].data, 0, NULL, &_buffer_ready_event);
		clEnqueueNDRangeKernel(_cl_queue, _cl_k_generate, 1, 0, &_buffer_size, &_workgroup_size, 0, NULL, NULL);
		clSetEventCallback(_buffer_ready_event, CL_COMPLETE, &__set_ready_flag, (void *) &_buffer32[_active_buffer32].ready);

		// switch active buffer
		_active_buffer32 = 1^_active_buffer32;
		_buffer32_i = 0;
	}
	return num;
}

cl_ulong __rand_gpu64()
{
	if (_buffer64_i == 0) {
		pthread_mutex_lock(&_buffer_ready_lock);
		while (!_buffer64[_active_buffer64].ready)
			pthread_cond_wait(&_buffer_ready_cond, &_buffer_ready_lock);
		pthread_mutex_unlock(&_buffer_ready_lock);
	}

	cl_ulong num = _buffer64[_active_buffer64].data[_buffer64_i++];

	// out of numbers in current buffer
	if (_buffer64_i == _buffer_size) {
		_buffer64[_active_buffer64].ready = CL_FALSE;
		// enqueue read data into the empty buffer, generate future numbers
		clEnqueueReadBuffer(_cl_queue, _cl_random_buf, CL_FALSE, 0, 
			_buffer_size * sizeof(cl_ulong), _buffer64[_active_buffer64].data, 0, NULL, &_buffer_ready_event);
		clEnqueueNDRangeKernel(_cl_queue, _cl_k_generate, 1, 0, &_buffer_size, &_workgroup_size, 0, NULL, NULL);
		clSetEventCallback(_buffer_ready_event, CL_COMPLETE, &__set_ready_flag, (void *) &_buffer64[_active_buffer64].ready);

		// switch active buffer
		_active_buffer64 = 1^_active_buffer64;
		_buffer64_i = 0;
	}
	return num;
}

void __clean_common()
{
    clFlush(_cl_queue);
	clFinish(_cl_queue);
	clReleaseMemObject(_cl_state_buf);
	clReleaseMemObject(_cl_random_buf);
	clReleaseKernel(_cl_k_init);
	clReleaseKernel(_cl_k_generate);
	clReleaseCommandQueue(_cl_queue);
	clReleaseProgram(_cl_program);
	clReleaseContext(_cl_context);
}

/*
	PUBLIC
*/

int rand_gpu32_init(uint32_t multi)
{
	return __gpu_init(32, multi);
}

int rand_gpu64_init(uint32_t multi)
{
	return __gpu_init(64, multi);
}

void rand_gpu64_clean()
{
	__clean_common();
	free(_buffer64[0].data);
	free(_buffer64[1].data);
}

void rand_gpu32_clean()
{
	__clean_common();
	free(_buffer32[0].data);
	free(_buffer32[1].data);
}

size_t rand_gpu_bufsiz() { return _buffer_size; }

int32_t  rand_gpu32_i32() { return __rand_gpu32(); }
int16_t  rand_gpu32_i16() { return __rand_gpu32(); }
uint32_t rand_gpu32_u32() { return __rand_gpu32(); }
uint16_t rand_gpu32_u16() { return __rand_gpu32(); }

int   rand_gpu32_int()   { return rand_gpu32_i32(); }
short rand_gpu32_short() { return rand_gpu32_i16(); }
unsigned int   rand_gpu32_uint()   { return rand_gpu32_u32(); }
unsigned short rand_gpu32_ushort() { return rand_gpu32_u16(); }

float rand_gpu32_float() { return FLOAT_MULTI_32 * __rand_gpu32(); }


int64_t  rand_gpu64_i64() { return __rand_gpu64(); }
int32_t  rand_gpu64_i32() { return __rand_gpu64(); }
int16_t  rand_gpu64_i16() { return __rand_gpu64(); }
uint64_t rand_gpu64_u64() { return __rand_gpu64(); }
uint32_t rand_gpu64_u32() { return __rand_gpu64(); }
uint16_t rand_gpu64_u16() { return __rand_gpu64(); }

long  rand_gpu64_long()  { return rand_gpu64_i64(); }
int   rand_gpu64_int()   { return rand_gpu64_i32(); }
short rand_gpu64_short() { return rand_gpu64_i16(); }
unsigned long  rand_gpu64_ulong()  { return rand_gpu64_u64(); }
unsigned int   rand_gpu64_uint()   { return rand_gpu64_u32(); }
unsigned short rand_gpu64_ushort() { return rand_gpu64_u16(); }

float  rand_gpu64_float()  { return FLOAT_MULTI_64  * __rand_gpu64(); }
double rand_gpu64_double() { return DOUBLE_MULTI_64 * __rand_gpu64(); }
