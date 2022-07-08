/**
@file

Implements RandomCL interface to tinymt64 RNG.

Tiny mersenne twister, http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/TINYMT/index.html.
*/


/**
 * TinyMT32 structure with parameters
 */
typedef struct TINYMT64WP_T {
    ulong s0;
    ulong s1;
    uint mat1;
    uint mat2;
    ulong tmat;
} tinymt64wp_t;

#define UINT64_C(x) (x ## UL)

#define TINYMT64J_MAT1 0xfa051f40U
#define TINYMT64J_MAT2 0xffd0fff4U;
#define TINYMT64J_TMAT UINT64_C(0x58d02ffeffbfffbc)

#define TINYMT64_SHIFT0 12
#define TINYMT64_SHIFT1 11
#define TINYMT64_MIN_LOOP 8

__constant ulong tinymt64_mask = 0x7fffffffffffffffUL;
__constant ulong tinymt64_double_mask = 0x3ff0000000000000UL;

/**
 * The function of the recursion formula calculation.
 *@param tiny internal state of tinymt with parameter
 */
inline static void
tinymt64_next_state(__global tinymt64wp_t * tiny)
{
    ulong x;

    tiny->s0 &= tinymt64_mask;
    x = tiny->s0 ^ tiny->s1;
    x ^= x << TINYMT64_SHIFT0;
    x ^= x >> 32;
    x ^= x << 32;
    x ^= x << TINYMT64_SHIFT1;
    tiny->s0 = tiny->s1;
    tiny->s1 = x;
    if (x & 1) {
        tiny->s0 ^= tiny->mat1;
        tiny->s1 ^= (ulong)tiny->mat2 << 32;
    }
}

/**
 * tempering output function
 *@param tiny internal state of tinymt with parameter
 *@return tempered output
 */
inline static ulong
tinymt64_temper(__global tinymt64wp_t * tiny)
{
    ulong x;
    x = tiny->s0 + tiny->s1;
    x ^= tiny->s0 >> 8;
    if (x & 1) {
        x ^= tiny->tmat;
    }
    return x;
}
/**
 * The function of the recursion formula calculation.
 *@param tiny internal state of tinymt with parameter
 *@return 32-bit random integer
 */
inline static ulong
tinymt64_uint64(__global tinymt64wp_t * tiny)
{
    tinymt64_next_state(tiny);
    return tinymt64_temper(tiny);
}

/**
 * Internal function.
 * This function certificate the period of 2^127-1.
 * @param tiny tinymt state vector.
 */
inline static void
tinymt64_period_certification(__global tinymt64wp_t * tiny)
{
    if ((tiny->s0 & tinymt64_mask) == 0 &&
        tiny->s1 == 0) {
        tiny->s0 = 'T';
        tiny->s1 = 'M';
    }
}

/**
 * This function initializes the internal state array with a 64-bit
 * unsigned integer seed.
 * @param tiny tinymt state vector.
 * @param seed a 64-bit unsigned integer used as a seed.
 */
inline static void
_tinymt64_init(__global tinymt64wp_t * tiny, ulong seed)
{
    ulong status[2];
    status[0] = seed ^ ((ulong)tiny->mat1 << 32);
    status[1] = tiny->mat2 ^ tiny->tmat;
    for (int i = 1; i < TINYMT64_MIN_LOOP; i++) {
        status[i & 1] ^= i + 6364136223846793005UL
            * (status[(i - 1) & 1] ^ (status[(i - 1) & 1] >> 62));
    }
    tiny->s0 = status[0];
    tiny->s1 = status[1];
    tinymt64_period_certification(tiny);
}

/**
State of tinymt64 RNG.
*/
typedef tinymt64wp_t tinymt64_state;


//#define tinymt64_seed(state, seed) tinymt64_init(state, seed)

/**
Seeds tinymt64 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void tinymt64_seed(__global tinymt64_state* state, ulong seed){
	state->mat1=TINYMT64J_MAT1;
	state->mat2=TINYMT64J_MAT2;
	state->tmat=TINYMT64J_TMAT;
	_tinymt64_init(state, seed);
}

__kernel void tinymt64_init(__global tinymt64_state *states, __global ulong *seeds)
{
    const uint gid = get_global_id(0);
    tinymt64_seed(&states[gid], seeds[gid]);
}

__kernel void tinymt64_generate(__global tinymt64_state *states, __global ulong *res)
{
    const uint gid = get_global_id(0);
    res[gid] = tinymt64_uint64(&states[gid]);
}
