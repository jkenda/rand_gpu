#pragma once
#include <stdio.h>

#define CL_TARGET_OPENCL_VERSION 210
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.h>
#endif

typedef struct
{
    cl_uint compute_units;
    cl_uint max_dimensions;
    size_t max_work_item_sizes[3];
    size_t max_work_group_sizes;
}
gpu_info_t;


void print_cl_err(cl_program program, cl_device_id device)
{
    size_t build_log_len;
    char *build_log;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);

    build_log = (char *) malloc(build_log_len + 1);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_len, build_log, NULL);
    printf("%s\n", build_log);
    free(build_log);
}

gpu_info_t gpu_info(cl_device_id device)
{
    int status = 0;
    gpu_info_t info;
    status += clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(info.compute_units), &info.compute_units, NULL);
    status += clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(info.compute_units), &info.max_dimensions, NULL);
    status += clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(info.compute_units), info.max_work_item_sizes, NULL);
    status += clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(info.max_work_group_sizes), &info.max_work_group_sizes, NULL);
    printf("status: %d\n", status);
    return info;
}
