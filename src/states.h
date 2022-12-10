#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

/**
State of kiss09 RNG.
*/
typedef struct {
	cl_ulong x,c,y,z;
} kiss09_state;

/**
State of lcg12864 RNG.
*/
typedef struct{
	cl_ulong low, high;
} lcg12864_state;

typedef char aligned_char __attribute__((aligned(8)));
#define LFIB_LAG1 17
/**
State of lfib RNG.
*/
typedef struct{
	cl_ulong s[LFIB_LAG1];
	aligned_char p1,p2; 
} lfib_state;

/**
State of mrg63k3a RNG.
*/
typedef struct{
	cl_long s10, s11, s12, s20, s21, s22;
} mrg63k3a_state;

/**
State of msws RNG.
*/
typedef struct{
	union{
		cl_ulong x;
		cl_uint2 x2;
	};
	cl_ulong w;
}msws_state;

typedef cl_int aligned_int __attribute__((aligned(8)));
#define MT19937_N 624
/**
State of MT19937 RNG.
*/
typedef struct __attribute__((aligned(16))){
	cl_uint mt[MT19937_N] __attribute__((aligned(8))); /* the array for the state vector  */
	aligned_int mti;
} mt19937_state;

/**
State of mwc64x RNG.
*/
typedef union {
	cl_ulong xc;
	struct { 
		cl_uint x;
		cl_uint c;
	} _;
} mwc64x_state;

/**
State of pcg6432 RNG.
*/
typedef  cl_ulong pcg6432_state;

/**
State of philox2x32_10 RNG.
*/
typedef union{
	cl_ulong LR;
	struct {
		cl_uint L, R;
	} _;
} philox2x32_10_state;

#define   NTAB 32
/**
State of ran2 RNG.
*/
typedef struct{
	cl_int idum;
	cl_int idum2;
	cl_int iy;
	cl_int iv[NTAB];
} ran2_state;

/**
 * TinyMT32 structure with parameters
 */
typedef struct TINYMT64WP_T {
    cl_ulong s0;
    cl_ulong s1;
    cl_uint mat1;
    cl_uint mat2;
    cl_ulong tmat;
} tinymt64wp_t;

/**
State of tyche RNG.
*/
typedef union{
	struct {
		cl_uint a,b,c,d;
	} _;
	cl_ulong res;
} tyche_state;

/**
State of tyche_i RNG.
*/
typedef union{
	struct {
		cl_uint a,b,c,d;
	} _;
	cl_ulong res;
} tyche_i_state;

#define R 16
/**
State of WELL RNG.
*/
typedef struct{
	cl_uint s[R];
	cl_uint i;
}well512_state;

/**
State of xorshift6432star RNG.
*/
typedef cl_ulong xorshift6432star_state;

