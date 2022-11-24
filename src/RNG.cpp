/**
 * @file RNG.cpp
 * @author Jakob Kenda (kenda.jakob@gmail.com)
 * @brief Source file for RNG/rand_gpu C/C++ library
 * @version 0.3
 * @date 2022-04-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "../include/RNG.hpp"
#include "../include/rand_gpu.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <array>
#include <unordered_set>
#include <map>
#include <mutex>
#include <condition_variable>
#include <random>
#include <iostream>
#include <cstring>
#include <atomic>
#include <chrono>
#include <future>

#define CL_MINIMUM_OPENCL_VERSION       100
#define CL_TARGET_OPENCL_VERSION        300
#define CL_HPP_MINIMUM_OPENCL_VERSION   100
#define CL_HPP_TARGET_OPENCL_VERSION    300
#define CL_HPP_ENABLE_EXCEPTIONS
#include "include/opencl.hpp"
#include "../kernel.hpp"
#include "states.h"

using namespace std;
using chrono::nanoseconds;
using chrono::system_clock;

atomic<nanoseconds> &operator+=(atomic<nanoseconds> &a, nanoseconds b)
{
    a.exchange(a.load() + b);
    return a;
}

static const map<rand_gpu_algorithm, const char *> NAMES = {
    { RAND_GPU_ALGORITHM_KISS09          ,    "KISS09"        },
    { RAND_GPU_ALGORITHM_LCG12864        ,    "LCG12864"      },
    { RAND_GPU_ALGORITHM_LFIB            ,    "LFIB"          },
    { RAND_GPU_ALGORITHM_MRG63K3A        ,    "MRG63K3A"      },
    { RAND_GPU_ALGORITHM_MSWS            ,    "MSWS"          },
    { RAND_GPU_ALGORITHM_MT19937         ,    "MT19937"       },
    { RAND_GPU_ALGORITHM_MWC64X          ,    "MWC64X"        },
    { RAND_GPU_ALGORITHM_PCG6432         ,    "PCG6432"       },
    { RAND_GPU_ALGORITHM_PHILOX2X32_10   ,    "PHILOX2X32_10" },
    { RAND_GPU_ALGORITHM_RAN2            ,    "RAN2"          },
    { RAND_GPU_ALGORITHM_TINYMT64        ,    "TINYMT64"      },
    { RAND_GPU_ALGORITHM_TYCHE           ,    "TYCHE"         },
    { RAND_GPU_ALGORITHM_TYCHE_I         ,    "TYCHE_I"       },
    { RAND_GPU_ALGORITHM_WELL512         ,    "WELL512"       },
    { RAND_GPU_ALGORITHM_XORSHIFT6432STAR,    "XORSHIFT6432*" },
};

static const map<rand_gpu_algorithm, const char *> DESCRIPTIONS = {
    { RAND_GPU_ALGORITHM_KISS09          , "Kiss09 - KISS (Keep It Simple, Stupid)"                                                                                     },
    { RAND_GPU_ALGORITHM_LCG12864        , "LCG12864 - 128-bit Linear Congruential Generator"                                                                           },
    { RAND_GPU_ALGORITHM_LFIB            , "LFib - Multiplicative Lagged Fibbonaci generator (lowest bit is always 1)"                                                  },
    { RAND_GPU_ALGORITHM_MRG63K3A        , "MRG63K3A (Multiple Recursive Generator)"                                                                                    },
    { RAND_GPU_ALGORITHM_MSWS            , "MSWS - Middle Square Weyl Sequence"                                                                                         },
    { RAND_GPU_ALGORITHM_MT19937         , "MT19937 - Mersenne twister: a 623-dimensionally equidistributed uniform pseudo-random number generator"                     },
    { RAND_GPU_ALGORITHM_MWC64X          , "MWC64x - 64-bit Multiply With Carry generator that returns 32-bit numbers that are xor of lower and upper 32-bit numbers"   },
    { RAND_GPU_ALGORITHM_PCG6432         , "PCG6432 - 64-bit Permutated Congruential generator (PCG-XSH-RR)"                                                            },
    { RAND_GPU_ALGORITHM_PHILOX2X32_10   , "Philox2x32_10"                                                                                                              },
    { RAND_GPU_ALGORITHM_RAN2            , "Ran2 - a L'Ecuyer combined recursive generator with a 32-element shuffle-box (from Numerical Recipes)"                      },
    { RAND_GPU_ALGORITHM_TINYMT64        , "Tinymt64 - Tiny Mersenne twister"                                                                                           },
    { RAND_GPU_ALGORITHM_TYCHE           , "Tyche - 512-bit Tyche (Well-Equidistributed Long-period Linear RNG)"                                                        },
    { RAND_GPU_ALGORITHM_TYCHE_I         , "Tyche-i - a faster variant of Tyche with a shorter period"                                                                  },
    { RAND_GPU_ALGORITHM_WELL512         , "WELL512 - 512-bit WELL (Well-Equidistributed Long-period Linear) RNG"                                                       },
    { RAND_GPU_ALGORITHM_XORSHIFT6432STAR, "xorshift6432* - 64-bit xorshift* generator that returns 32-bit values"                                                      },
};


static const map<rand_gpu_algorithm, size_t> STATE_SIZES = {
    { RAND_GPU_ALGORITHM_KISS09          , sizeof(kiss09_state)             },
    { RAND_GPU_ALGORITHM_LCG12864        , sizeof(lcg12864_state)           },
    { RAND_GPU_ALGORITHM_LFIB            , sizeof(lfib_state)               },
    { RAND_GPU_ALGORITHM_MRG63K3A        , sizeof(mrg63k3a_state)           },
    { RAND_GPU_ALGORITHM_MSWS            , sizeof(msws_state)               },
    { RAND_GPU_ALGORITHM_MT19937         , sizeof(mt19937_state)            },
    { RAND_GPU_ALGORITHM_MWC64X          , sizeof(mwc64x_state)             },
    { RAND_GPU_ALGORITHM_PCG6432         , sizeof(pcg6432_state)            },
    { RAND_GPU_ALGORITHM_PHILOX2X32_10   , sizeof(philox2x32_10_state)      },
    { RAND_GPU_ALGORITHM_RAN2            , sizeof(ran2_state)               },
    { RAND_GPU_ALGORITHM_TINYMT64        , sizeof(tinymt64wp_t)             },
    { RAND_GPU_ALGORITHM_TYCHE           , sizeof(tyche_state)              },
    { RAND_GPU_ALGORITHM_TYCHE_I         , sizeof(tyche_i_state)            },
    { RAND_GPU_ALGORITHM_WELL512         , sizeof(well512_state)            },
    { RAND_GPU_ALGORITHM_XORSHIFT6432STAR, sizeof(xorshift6432star_state)   },
};

static const map<rand_gpu_algorithm, const char *> INIT_KERNEL_NAMES = {
    { RAND_GPU_ALGORITHM_KISS09          , "kiss09_init"            },
    { RAND_GPU_ALGORITHM_LCG12864        , "lcg12864_init"          },
    { RAND_GPU_ALGORITHM_LFIB            , "lfib_init"              },
    { RAND_GPU_ALGORITHM_MRG63K3A        , "mrg63k3a_init"          },
    { RAND_GPU_ALGORITHM_MSWS            , "msws_init"              },
    { RAND_GPU_ALGORITHM_MT19937         , "mt19937_init"           },
    { RAND_GPU_ALGORITHM_MWC64X          , "mwc64x_init"            },
    { RAND_GPU_ALGORITHM_PCG6432         , "pcg6432_init"           },
    { RAND_GPU_ALGORITHM_PHILOX2X32_10   , "philox2x32_10_init"     },
    { RAND_GPU_ALGORITHM_RAN2            , "ran2_init"              },
    { RAND_GPU_ALGORITHM_TINYMT64        , "tinymt64_init"          },
    { RAND_GPU_ALGORITHM_TYCHE           , "tyche_init"             },
    { RAND_GPU_ALGORITHM_TYCHE_I         , "tyche_i_init"           },
    { RAND_GPU_ALGORITHM_WELL512         , "well512_init"           },
    { RAND_GPU_ALGORITHM_XORSHIFT6432STAR, "xorshift6432star_init"  },
};

static const map<rand_gpu_algorithm, const char *> GENERATE_KERNEL_NAMES = {
    { RAND_GPU_ALGORITHM_KISS09          , "kiss09_generate"            },
    { RAND_GPU_ALGORITHM_LCG12864        , "lcg12864_generate"          },
    { RAND_GPU_ALGORITHM_LFIB            , "lfib_generate"              },
    { RAND_GPU_ALGORITHM_MRG63K3A        , "mrg63k3a_generate"          },
    { RAND_GPU_ALGORITHM_MSWS            , "msws_generate"              },
    { RAND_GPU_ALGORITHM_MT19937         , "mt19937_generate"           },
    { RAND_GPU_ALGORITHM_MWC64X          , "mwc64x_generate"            },
    { RAND_GPU_ALGORITHM_PCG6432         , "pcg6432_generate"           },
    { RAND_GPU_ALGORITHM_PHILOX2X32_10   , "philox2x32_10_generate"     },
    { RAND_GPU_ALGORITHM_RAN2            , "ran2_generate"              },
    { RAND_GPU_ALGORITHM_TINYMT64        ,  "tinymt64_generate"         },
    { RAND_GPU_ALGORITHM_TYCHE           ,  "tyche_generate"            },
    { RAND_GPU_ALGORITHM_TYCHE_I         ,  "tyche_i_generate"          },
    { RAND_GPU_ALGORITHM_WELL512         ,  "well512_generate"          },
    { RAND_GPU_ALGORITHM_XORSHIFT6432STAR,  "xorshift6432star_generate" },
};

static const float       FLOAT_MULTIPLIER       = 1 /       (float) UINT32_MAX;
static const double      DOUBLE_MULTIPLIER      = 1 /      (double) UINT64_MAX;
static const long double LONG_DOUBLE_MULTIPLIER = 1 / (long double) UINT64_MAX;
static const float       MILLION                = (float) 1'000'000;
static const size_t      N_ALGORITHMS           = RAND_GPU_ALGORITHM_XORSHIFT6432STAR - RAND_GPU_ALGORITHM_KISS09 + 1;


static mutex __constructor_lock;
static bool __initialized = false;

static atomic<size_t> __memory_usage = 0;

static mt19937_64 __generator;
static vector<cl::Device> __devices;
static array<map<cl::Device *, cl::Program>, N_ALGORITHMS> __programs;
static array<nanoseconds, N_ALGORITHMS> __compilation_times;
static map<cl::Device *, cl::Context> __context;
static size_t __device_id = 0;

static atomic<nanoseconds> __sum_init_time_deleted = nanoseconds::zero();
static atomic<nanoseconds> __sum_avg_transfer_time_deleted = nanoseconds::zero();
static atomic<nanoseconds> __sum_avg_calculation_time_deleted = nanoseconds::zero();

static atomic<size_t> __sum_buffer_misses_deleted = 0;
static atomic<size_t> __sum_buffer_switches_deleted = 0;
static atomic<size_t> __n_deleted_rngs = 0;

static mutex __rngs_lock;
static mutex __delete_all_lock;
static unordered_set<RNG_impl *> __rngs;


struct RNG_impl
{
    struct Buffer
    {
        cl::Kernel k_generate;                          // CL kernel that generates pseudo-random numbers

        cl::Buffer device;                              // device buffer
        uint8_t *host = nullptr;                        // host buffer

        future<void> ready = async([]{ return; });                      // flag signifies whether buffer has been filled
        cl::Event transferred_event, calculated_event;  // CL events for used for callbacks

        system_clock::time_point start_time;            // time point when kernel was enqueued
        RNG_impl *rng;                                  // RNG this buffer is part of
    };

    cl::CommandQueue _queue;        // command queue for RNG instance
    cl::Buffer       _state_buf;    // buffer containing RNGs' internal states

    cl::NDRange _global_size;       // kernel's global size -- number of RNGs on GPU
    size_t      _buffer_size;       // random number buffer size -- (_global_size * 8) bytes
    size_t      _buf_offset_limit;  // maximal value of _buf_offset -- _buffer_size - 8
    bool        _unified_memory;    // are CPU and GPU using the same memory?

    const size_t   _n_buffers;          // number of buffers
    vector<Buffer> _buffers;            // vector of buffers
    uint_fast8_t   _active_buf_id = 0;  // which buffer is currently being read
    uint_fast64_t  _buf_offset = 0;     // offset into current buffer

    uint64_t      _bool_reg;            // register that enables usage of every single bit for boolean values
    uint_fast8_t  _bool_reg_offset = 0; // offset into the bool register

    nanoseconds _init_time = nanoseconds::zero();                   // initialization time
    nanoseconds _gpu_calculation_time_total = nanoseconds::zero();  // sum of all calculation times
    nanoseconds _gpu_transfer_time_total = nanoseconds::zero();     // sum of all transfer times

    atomic<size_t> _n_gpu_transfers = 0;    // number of GPU transfers
    atomic<size_t> _n_gpu_calculations = 0; // number or GPU calculations
    atomic<size_t> _n_buffer_switches = 0;  // number of buffer switches
    atomic<size_t> _n_buffer_misses = 0;    // number of buffer misses (times CPU to wait for GPU)


    RNG_impl(size_t n_buffers = 2, size_t multi = 1, rand_gpu_algorithm algorithm = RAND_GPU_ALGORITHM_TYCHE,
        bool use_custom_seed = false, uint64_t custom_seed = 0)
    :
        _n_buffers(max(n_buffers, 2LU)),
        _buffers(max(n_buffers, 2LU))
    {
        uint64_t seed = 0;
        auto start = chrono::system_clock::now();
        size_t device_id = 0;

        // ENTER LOCK
        __constructor_lock.lock();

        if (!__initialized)
        {
            cl::Platform platform;

            // get platforms and devices
            try
            {
                cl::Platform::get(&platform);
                platform.getDevices(CL_DEVICE_TYPE_GPU, &__devices);
            }
            catch (const cl::Error &err)
            {
                cerr << "No openCL platforms/devices found!\n";
                cerr << "Forgot 'srun'?\n";
                throw err;
            }

            // initialize host RNG for generating seeds
            random_device dev;
            uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
            __generator = mt19937_64(dist(dev));

            __initialized = true;
        }

        // generate seed
        if (use_custom_seed)
            seed = custom_seed;
        else
            seed = __generator();

        // distribute devices among instances
        device_id = __device_id;
        __device_id = (__device_id + 1) % __devices.size();

        // create context and program
        cl::Device &device   = __devices[device_id];
        bool program_built   = __programs[algorithm].count(&device) > 0;
        cl::Program &program = __programs[algorithm][&device];
        auto [it, _] = __context.try_emplace(&device, device);
        cl::Context &context = it->second;

        if (!program_built)
        {
            program = cl::Program(context, KERNEL_SOURCE[algorithm]);

            try
            {
                auto build_start = system_clock::now();
                program.build(device);
                __compilation_times[algorithm] = system_clock::now() - build_start;
            }
            catch (const cl::Error &err)
            {
                // print buildlog if build failed
                if (program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) != CL_SUCCESS)
                {
                    string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                    cerr << buildlog << '\n';
                    throw err;
                }
            }
        }

        // EXIT LOCK
        __constructor_lock.unlock();


        // create command queue
        _queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

        // get device info
        cl::Kernel temp(program, INIT_KERNEL_NAMES.at(algorithm));
        _unified_memory        = device.getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>();
        size_t max_cu          = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        size_t maximal_wg_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        size_t optimal_wg_size = temp.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);

        // calculate optimal buffer size
        size_t local_size  = min(optimal_wg_size, maximal_wg_size);
        size_t global_size = max(multi, 1UL) * max_cu * local_size;
        _buffer_size = global_size * sizeof(uint64_t);
        _global_size = cl::NDRange(_buffer_size / sizeof(uint64_t));
        _buf_offset_limit = _buffer_size - sizeof(uint64_t);

        // create state buffer
        size_t state_buf_size = _global_size[0] * STATE_SIZES.at(algorithm);
        _state_buf = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, state_buf_size);

        // initialize RNG
        cl::Kernel k_init(program, INIT_KERNEL_NAMES.at(algorithm));
        cl::Event initialized;
        k_init.setArg(0, _state_buf);

        vector<uint64_t> seeds(0);
        cl::Buffer cl_seeds;

        switch (algorithm)
        {
        case RAND_GPU_ALGORITHM_TYCHE:
        case RAND_GPU_ALGORITHM_TYCHE_I:
            // a single seed for all threads
            k_init.setArg(1, sizeof(uint64_t), &seed);
            break;
        case RAND_GPU_ALGORITHM_MT19937:
            // 32-bit seeds
            {
                mt19937_64 seed_generator(seed);
                seeds.resize(_global_size[0] / 2);
                for (auto& seed : seeds)
                    seed = seed_generator();
                cl_seeds = cl::Buffer(context, seeds.begin(), seeds.end(), true, _unified_memory);
                k_init.setArg(1, cl_seeds);
            }
            break;
        default:
            // 64-bit seeds
            {
                mt19937_64 seed_generator(seed);
                seeds.resize(_global_size[0]);
                for (auto& seed : seeds)
                    seed = seed_generator();
                cl_seeds = cl::Buffer(context, seeds.begin(), seeds.end(), true, _unified_memory);
                k_init.setArg(1, cl_seeds);
            }
        }

        // initialize RNGs on GPU
        _queue.enqueueNDRangeKernel(k_init, cl::NullRange, _global_size);


        // allocate and map device and host buffers
        for (Buffer &buffer : _buffers)
        {
            // let OpenCL library allocate host buffer (CL_MEM_ALLOC_HOST_PTR)
            buffer.device = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, _buffer_size);
            buffer.host = (uint8_t *) _queue.enqueueMapBuffer(buffer.device, true, CL_MAP_READ, 0, _buffer_size);
            buffer.rng = this;
        }

        // create kernels for buffers
        for (Buffer &buffer : _buffers)
        {
            buffer.k_generate = cl::Kernel(program, GENERATE_KERNEL_NAMES.at(algorithm));
            buffer.k_generate.setArg(0, _state_buf);
            buffer.k_generate.setArg(1, buffer.device);
        }

        // generate 1st batch of numbers
        for (Buffer &buffer : _buffers)
        {
            if (_unified_memory)
            {
                _queue.enqueueNDRangeKernel(buffer.k_generate, 0, _global_size, cl::NullRange);
            }
            else
            {
                _queue.enqueueNDRangeKernel(buffer.k_generate, cl::NullRange, _global_size);
                _queue.enqueueReadBuffer(buffer.device, false, 0, _buffer_size, buffer.host);
                _queue.enqueueNDRangeKernel(buffer.k_generate, cl::NullRange, _global_size);
            }
        }

        // wait for tasks in queue to finish
        _queue.finish();

        // initialize the bool register
        _bool_reg = get_random<uint64_t>();

        // update stats
        _init_time = chrono::system_clock::now() - start;
        __memory_usage += _n_buffers * _buffer_size;

        lock_guard<mutex> rngs_lock(__rngs_lock);
        __rngs.insert(this);
    }

    ~RNG_impl()
    {
        for (Buffer &buffer : _buffers)
        {
            // wait for the OpenCL thread to finish work
            buffer.ready.wait();

            // unmap host_ptr
            _queue.enqueueUnmapMemObject(buffer.device, buffer.host);
        }

        _queue.flush();
        _queue.finish();

        // update stats
        __memory_usage -= _n_buffers * _buffer_size;
        __sum_avg_transfer_time_deleted    += avg_gpu_transfer_time();
        __sum_avg_calculation_time_deleted += avg_gpu_calculation_time();
        __sum_init_time_deleted            += _init_time;
        __sum_buffer_switches_deleted      += _n_buffer_switches;
        __sum_buffer_misses_deleted        += _n_buffer_misses;
        __n_deleted_rngs++;

        lock_guard<mutex> rngs_lock(__rngs_lock);
        __rngs.erase(this);
    }

    void fill_buffer(Buffer &buffer)
    {
        // reading buffers is not neccessary if the GPU is integrated
        if (_unified_memory)
        {
            // calculate next batch of numbers
            _queue.enqueueNDRangeKernel(buffer.k_generate, 0, _global_size, cl::NullRange,
                                        NULL, &buffer.calculated_event);

            auto start_time = system_clock::now();
            buffer.calculated_event.wait();
            auto end_time = system_clock::now();

            // get calculation time
            uint64_t calc_start = buffer.calculated_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            uint64_t calc_end   = buffer.calculated_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

            nanoseconds calc_time(calc_end - calc_start);

            // update calculation and transfer time
            _gpu_calculation_time_total += calc_time;
            _gpu_transfer_time_total    += end_time - start_time - calc_time;
            _n_gpu_transfers++;
            _n_gpu_calculations++;
        }
        else
        {

            // read calculated numbers
            _queue.enqueueReadBuffer(buffer.device, false, 0, _buffer_size, buffer.host,
                                     NULL, &buffer.transferred_event);

            auto start_time = system_clock::now();
            // calculate next batch
            _queue.enqueueNDRangeKernel(buffer.k_generate, cl::NullRange, _global_size, cl::NullRange,
                                        NULL, &buffer.calculated_event);

            buffer.transferred_event.wait();
            auto end_time = system_clock::now();

            // update transfer time
            _gpu_transfer_time_total += end_time - start_time;
            _n_gpu_transfers++;

            buffer.calculated_event.wait();

            // get calculation time
            uint64_t calc_start = buffer.calculated_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            uint64_t calc_end   = buffer.calculated_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            nanoseconds calc_time(calc_end - calc_start);

            _gpu_calculation_time_total += calc_time;
            _n_gpu_calculations++;
        }

    }

    inline __attribute__((always_inline)) void switch_and_fill_buffer(Buffer &buffer)
    {
        buffer.ready = async(launch::async, [&]{ fill_buffer(buffer); });

        // switch active buffer
        _active_buf_id = (_active_buf_id + 1) % _n_buffers;
        _buf_offset = 0;
        _n_buffer_switches++;
    }


    template <typename T>
    inline __attribute__((always_inline)) T get_random()
    {
        Buffer &active_buf = _buffers[_active_buf_id];

        // just switched buffers - wait for buffer to be ready
        active_buf.ready.wait();

        // retrieve number from buffer
        T num;
        memcpy(&num, active_buf.host + _buf_offset, sizeof(T));
        _buf_offset += sizeof(T);

        // out of numbers in current buffer
        if (_buf_offset >= _buf_offset_limit)
            switch_and_fill_buffer(active_buf);

        return num;
    }

    inline __attribute__((always_inline)) void put_random(void *dst, const size_t &nbytes)
    {
        Buffer *active_buf = &_buffers[_active_buf_id];

        // just switched buffers - wait for buffer to be ready
        if (_buf_offset == 0)
        {
            active_buf->ready.wait();
        }
        else if (_buf_offset + nbytes > _buffer_size)
        {
            // copy to the end of the buffer
            size_t transferred_bytes = _buffer_size - _buf_offset;
            memcpy(dst, &active_buf->host[_buf_offset], transferred_bytes);
            size_t transfer_size = 0;

            // copy the rest
            while (transferred_bytes < nbytes)
            {
                switch_and_fill_buffer(*active_buf);
                active_buf = &_buffers[_active_buf_id];
                transfer_size = min(nbytes - transferred_bytes, _buffer_size);
                memcpy((uint8_t *) dst + transferred_bytes, active_buf->host, transfer_size);
                transferred_bytes += transfer_size;
            }
            _buf_offset = transfer_size;
        }
        else
        {
            // copy the required number of bytes
            memcpy(dst, &active_buf->host[_buf_offset], nbytes);
            _buf_offset += nbytes;
        }
    }

    void discard(size_t z)
    {
        // discard z bytes
        _buf_offset += z;

        // out of numbers in current buffer
        if (_buf_offset >= _buf_offset_limit)
            switch_and_fill_buffer(_buffers[_active_buf_id]);
    }

    nanoseconds avg_gpu_calculation_time() const
    {
        if (_n_gpu_calculations > 0)
            return _gpu_calculation_time_total / _n_gpu_calculations.load();
        else
            return chrono::nanoseconds::zero();
    }

    nanoseconds avg_gpu_transfer_time() const
    {
        if (_n_gpu_transfers > 0)
            return _gpu_transfer_time_total / _n_gpu_transfers.load();
        else
            return chrono::nanoseconds::zero();
    }

};


/*
template specialization
*/

template <>
inline __attribute__((always_inline)) float RNG_impl::get_random<float>()
{
    // map to [0, 1) -- multiply by 1 / UINT32_MAX
    return FLOAT_MULTIPLIER * get_random<uint32_t>();
}

template <>
inline __attribute__((always_inline)) double RNG_impl::get_random<double>()
{
    // map to [0, 1) -- multiply by 1 / UINT64_MAX
    return DOUBLE_MULTIPLIER * get_random<uint64_t>();
}

template <>
inline __attribute__((always_inline)) long double RNG_impl::get_random<long double>()
{
    // map to [0, 1) -- multiply by 1 / UINT64_MAX
    return LONG_DOUBLE_MULTIPLIER * get_random<uint64_t>();
}

template <>
inline __attribute__((always_inline)) bool RNG_impl::get_random<bool>()
{
    // convert a bit in _bool_register to bool
    const bool from_bit = (_bool_reg >> _bool_reg_offset++) & 1;

    // out of bits in register
    if (_bool_reg_offset > sizeof(uint64_t))
    {
        _bool_reg = get_random<uint64_t>();
        _bool_reg_offset = 0;
    }

    return from_bit;
}


/*
C wrapper functions
*/

extern "C" {

rand_gpu_rng rand_gpu_new_rng_default()
{
    return (rand_gpu_rng ) new RNG_impl;
}

rand_gpu_rng rand_gpu_new_rng(enum rand_gpu_algorithm algorithm, size_t n_buffers, size_t buffer_multi)
{
    return (rand_gpu_rng ) new RNG_impl(n_buffers, buffer_multi, algorithm);
}

rand_gpu_rng rand_gpu_new_rng_with_seed(uint64_t seed, enum rand_gpu_algorithm algorithm, size_t n_buffers, size_t buffer_multi)
{
    return (rand_gpu_rng ) new RNG_impl(n_buffers, buffer_multi, algorithm, true, seed);
}

void rand_gpu_delete_rng(rand_gpu_rng rng)
{
    delete (RNG_impl *) rng;
}

void rand_gpu_delete_all()
{
    lock_guard<mutex> lock(__delete_all_lock);
    for (auto it = __rngs.begin(); it != __rngs.end();)
    {
        delete *(it++);
    }
}

void rand_gpu_rng_discard(rand_gpu_rng rng, uint64_t z) { ((RNG_impl *) rng)->discard(z); }

uint64_t rand_gpu_rng_get_random_64b(rand_gpu_rng rng) { return ((RNG_impl *) rng)->get_random<uint64_t>(); }
uint32_t rand_gpu_rng_get_random_32b(rand_gpu_rng rng) { return ((RNG_impl *) rng)->get_random<uint32_t>(); }
uint16_t rand_gpu_rng_get_random_16b(rand_gpu_rng rng) { return ((RNG_impl *) rng)->get_random<uint16_t>(); }
uint8_t  rand_gpu_rng_get_random_8b(rand_gpu_rng rng)  { return ((RNG_impl *) rng)->get_random<uint8_t>();  }

uint8_t     rand_gpu_get_random_bool(rand_gpu_rng rng)        { return ((RNG_impl *) rng)->get_random<bool>();  }
float       rand_gpu_get_random_float(rand_gpu_rng rng)       { return ((RNG_impl *) rng)->get_random<float>();       }
double      rand_gpu_get_random_double(rand_gpu_rng rng)      { return ((RNG_impl *) rng)->get_random<double>();      }
long double rand_gpu_get_random_long_double(rand_gpu_rng rng) { return ((RNG_impl *) rng)->get_random<long double>(); }

void rand_gpu_rng_put_random(rand_gpu_rng rng, void *dst, const size_t nbytes) { ((RNG_impl *) rng)->put_random(dst, nbytes); }

size_t rand_gpu_rng_buffer_size(const rand_gpu_rng rng)              { return ((const RNG_impl *) rng)->_buffer_size; }
size_t rand_gpu_rng_buffer_switches(const rand_gpu_rng rng)          { return ((const RNG_impl *) rng)->_n_buffer_switches; }
size_t rand_gpu_rng_buffer_misses(const rand_gpu_rng rng)            { return ((const RNG_impl *) rng)->_n_buffer_misses; }
float  rand_gpu_rng_init_time(const rand_gpu_rng rng)                { return ((const RNG_impl *) rng)->_init_time.count() / MILLION; }
float  rand_gpu_rng_avg_gpu_calculation_time(const rand_gpu_rng rng) { return ((const RNG_impl *) rng)->avg_gpu_calculation_time().count() / MILLION; }
float  rand_gpu_rng_avg_gpu_transfer_time(const rand_gpu_rng rng)    { return ((const RNG_impl *) rng)->avg_gpu_transfer_time().count() / MILLION; }

size_t rand_gpu_memory_usage()             { return __memory_usage; }
size_t rand_gpu_buffer_switches()          { return rand_gpu::buffer_switches(); }
size_t rand_gpu_buffer_misses()            { return rand_gpu::buffer_misses(); }
float  rand_gpu_avg_init_time()            { return rand_gpu::avg_init_time().count() / MILLION; }
float  rand_gpu_avg_gpu_calculation_time() { return rand_gpu::avg_gpu_calculation_time().count() / MILLION; }
float  rand_gpu_avg_gpu_transfer_time()    { return rand_gpu::avg_gpu_transfer_time().count() / MILLION; }

float rand_gpu_compilation_time(rand_gpu_algorithm algorithm) { return __compilation_times[algorithm].count() / MILLION; }
const char *rand_gpu_algorithm_name(rand_gpu_algorithm algorithm, bool description) { return rand_gpu::algorithm_name(algorithm, description); }

}


/*
RNG definitions
*/

namespace rand_gpu
{
    template <rand_gpu_algorithm A>
    RNG<A>::RNG()
    :
        _impl_ptr(new RNG_impl(2, 1, A, false))
    {
    }

    template <rand_gpu_algorithm A>
    RNG<A>::RNG(rand_gpu_rng rng)
    :
        _impl_ptr((RNG_impl *) rng)
    {
    }

    template <rand_gpu_algorithm A>
    RNG<A>::RNG(size_t n_buffers, size_t multi)
    :
        _impl_ptr(new RNG_impl(n_buffers, multi, A, false))
    {
    }

    template <rand_gpu_algorithm A>
    RNG<A>::RNG(unsigned long seed, size_t n_buffers, size_t multi)
    :
        _impl_ptr(new RNG_impl(n_buffers, multi, A, true, seed))
    {
    }

    template <rand_gpu_algorithm A>
    RNG<A>::~RNG()
    {
        delete _impl_ptr;
    }

    template <rand_gpu_algorithm A>
    RNG<A>::RNG(RNG&&) = default;
    template <rand_gpu_algorithm A>
    RNG<A> &RNG<A>::operator=(RNG&&) = default;

    template <rand_gpu_algorithm A>
    template <typename T>
    T RNG<A>::get_random()
    {
        return _impl_ptr->get_random<T>();
    }

    template <rand_gpu_algorithm A>
    void RNG<A>::put_random(void *dst, const size_t &nbytes)
    {
        return _impl_ptr->put_random(dst, nbytes);
    }

    template <rand_gpu_algorithm A>
    uint64_t RNG<A>::operator()()
    {
       return _impl_ptr->get_random<uint64_t>(); 
    }

    template <rand_gpu_algorithm A>
    void RNG<A>::discard(size_t z)
    {
        _impl_ptr->discard(z);
    }

    template <rand_gpu_algorithm A>
    size_t RNG<A>::buffer_size() const
    {
        return _impl_ptr->_buffer_size;
    }

    template <rand_gpu_algorithm A>
    size_t RNG<A>::buffer_switches() const
    {
        return _impl_ptr->_n_buffer_switches;
    }

    template <rand_gpu_algorithm A>
    size_t RNG<A>::buffer_misses() const
    {
        return _impl_ptr->_n_buffer_misses;
    }

    template <rand_gpu_algorithm A>
    nanoseconds RNG<A>::init_time() const
    {
        return _impl_ptr->_init_time;
    }

    template <rand_gpu_algorithm A>
    std::chrono::nanoseconds RNG<A>::avg_gpu_calculation_time() const
    {
        return _impl_ptr->avg_gpu_calculation_time();
    }

    template <rand_gpu_algorithm A>
    std::chrono::nanoseconds RNG<A>::avg_gpu_transfer_time() const
    {
        return _impl_ptr->avg_gpu_transfer_time();
    }

    size_t memory_usage()
    {
        return __memory_usage;
    }


    nanoseconds avg_gpu_calculation_time()
    {
        nanoseconds sum(0);
        lock_guard<mutex> lock(__rngs_lock);

        // add the averages of active RNGs
        for (RNG_impl *rng : __rngs)
        {
            sum += rng->avg_gpu_calculation_time();
        }

        // add the averages of deleted RNGs
        sum += __sum_avg_calculation_time_deleted.load();
        return sum / (__rngs.size() + __n_deleted_rngs);
    }

    nanoseconds avg_gpu_transfer_time()
    {
        nanoseconds sum(0);
        lock_guard<mutex> lock(__rngs_lock);

        // add the averages of active RNGs
        for (RNG_impl *rng : __rngs)
        {
            sum += rng->avg_gpu_transfer_time();
        }

        // add the averages of deleted RNGs
        sum += __sum_avg_transfer_time_deleted.load();
        return sum / (__rngs.size() + __n_deleted_rngs);
    }

    nanoseconds avg_init_time()
    {
        nanoseconds sum(0);
        lock_guard<mutex> lock(__rngs_lock);

        // add the averages of active RNGs
        for (RNG_impl *rng : __rngs)
        {
            sum += rng->_init_time;
        }

        // add the averages of deleted RNGs
        sum += __sum_init_time_deleted.load();
        return sum / (__rngs.size() + __n_deleted_rngs);
    }

    size_t buffer_switches()
    {
        size_t sum = 0;
        lock_guard<mutex> lock(__rngs_lock);

        // add the averages of active RNGs
        for (RNG_impl *rng : __rngs)
        {
            sum += rng->_n_buffer_switches;
        }

        // add the averages of deleted RNGs
        sum += __sum_buffer_switches_deleted;
        return sum;
    }

    size_t buffer_misses()
    {
        size_t sum = 0;
        lock_guard<mutex> lock(__rngs_lock);

        // add the averages of active RNGs
        for (RNG_impl *rng : __rngs)
        {
            sum += rng->_n_buffer_misses;
        }

        // add the averages of deleted RNGs
        sum += __sum_buffer_misses_deleted;
        return sum;
    }

    std::chrono::nanoseconds compilation_time(rand_gpu_algorithm algorithm)
    {
        return __compilation_times[algorithm];
    }

    const char *algorithm_name(rand_gpu_algorithm algorithm, bool description)
    {
        return description ? DESCRIPTIONS.at(algorithm) : NAMES.at(algorithm);
    }

    /*
    instantiate templates for all algorithms and primitives
    */

    template class RNG<RAND_GPU_ALGORITHM_KISS09>;
    template class RNG<RAND_GPU_ALGORITHM_LCG12864>;
    template class RNG<RAND_GPU_ALGORITHM_LFIB>;
    template class RNG<RAND_GPU_ALGORITHM_MRG63K3A>;
    template class RNG<RAND_GPU_ALGORITHM_MSWS>;
    template class RNG<RAND_GPU_ALGORITHM_MT19937>;
    template class RNG<RAND_GPU_ALGORITHM_MWC64X>;
    template class RNG<RAND_GPU_ALGORITHM_PCG6432>;
    template class RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>;
    template class RNG<RAND_GPU_ALGORITHM_RAN2>;
    template class RNG<RAND_GPU_ALGORITHM_TINYMT64>;
    template class RNG<RAND_GPU_ALGORITHM_TYCHE>;
    template class RNG<RAND_GPU_ALGORITHM_TYCHE_I>;
    template class RNG<RAND_GPU_ALGORITHM_WELL512>;
    template class RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>;

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_KISS09>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_LCG12864>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_LFIB>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_MRG63K3A>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_MSWS>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_MT19937>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_MWC64X>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_PCG6432>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_PHILOX2X32_10>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_RAN2>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_TINYMT64>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_TYCHE>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_TYCHE_I>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_WELL512>::get_random<long double>();

    template unsigned long long int RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<unsigned long long int>();
    template unsigned long int  RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<unsigned long int>();
    template unsigned int       RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<unsigned int>();
    template unsigned short int RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<unsigned short int>();
    template unsigned char      RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<unsigned char>();
    template long long int RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<long long int>();
    template long int      RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<long int>();
    template int           RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<int>();
    template short int     RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<short int>();
    template signed char   RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<signed char>();
    template char RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<char>();
    template bool RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<bool>();
    template float       RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<float>();
    template double      RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<double>();
    template long double RNG<RAND_GPU_ALGORITHM_XORSHIFT6432STAR>::get_random<long double>();

}
