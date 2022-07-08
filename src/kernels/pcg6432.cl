/**
@file

Implements a 64-bit Permutated Congruential generator (PCG-XSH-RR).

M. E. O’Neill, Pcg: A family of simple fast space-efficient statistically good algorithms for random number generation, ACM Transactions on Mathematical Software.
*/


/**
State of pcg6432 RNG.
*/
typedef unsigned long pcg6432_state;

#define PCG6432_XORSHIFTED(s) ((uint)((((s) >> 18u) ^ (s)) >> 27u))
#define PCG6432_ROT(s) ((s) >> 59u)

#define pcg6432_macro_uint(state) ( \
	state = state * 6364136223846793005UL + 0xda3e39cb94b95bdbUL, \
	(PCG6432_XORSHIFTED(state) >> PCG6432_ROT(state)) | (PCG6432_XORSHIFTED(state) << ((-PCG6432_ROT(state)) & 31)) \
)

/**
Generates a random 32-bit unsigned integer using pcg6432 RNG.

@param state State of the RNG to use.
*/
#define pcg6432_uint(state) _pcg6432_uint(&state)
unsigned int _pcg6432_uint(__global pcg6432_state* state){
    ulong oldstate = *state;
	*state = oldstate * 6364136223846793005UL + 0xda3e39cb94b95bdbUL;
	uint xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
	uint rot = oldstate >> 59u;
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

/**
Seeds pcg6432 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void pcg6432_seed(__global pcg6432_state* state, unsigned long j){
	*state=j;
}

/**
Generates a random 64-bit unsigned integer using pcg6432 RNG.

@param state State of the RNG to use.
*/
#define pcg6432_ulong(state) ((((ulong)pcg6432_uint(state)) << 32) | pcg6432_uint(state))

__kernel void pcg6432_init(__global pcg6432_state *states, __global ulong *seeds)
{
    const uint gid = get_global_id(0);
    pcg6432_seed(&states[gid], seeds[gid]);
}

__kernel void pcg6432_generate(__global pcg6432_state *states, __global ulong *res)
{
    const uint gid = get_global_id(0);
    res[gid] = pcg6432_ulong(states[gid]);
}
