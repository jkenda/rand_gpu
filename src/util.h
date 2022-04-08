#pragma once
#include <stdio.h>

#define CL_TARGET_OPENCL_VERSION 210
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.h>
#endif


void print_cl_err(cl_program program, cl_device_id device_id)
{
    size_t build_log_len;
    char *build_log;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);

    build_log = (char *) malloc(build_log_len + 1);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, build_log_len, build_log, NULL);
    printf("%s\n", build_log);
    free(build_log);
}
