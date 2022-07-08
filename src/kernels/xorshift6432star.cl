/**
@file

Implements a 64-bit xorshift* generator that returns 32-bit values.

S. Vigna, An experimental exploration of marsaglia’s xorshift generators, scrambled, ACM Transactions on Mathematical Software (TOMS) 42 (4) (2016) 30.
*/


/**
State of xorshift6432star RNG.
*/
typedef unsigned long xorshift6432star_state;

#define xorshift6432star_macro_uint(state) (\
	state ^= state >> 12, \
	state ^= state << 25, \
	state ^= state >> 27, \
	(uint)((state*0x2545F4914F6CDD1D)>>32) \
	)

/**
Generates a random 32-bit unsigned integer using xorshift6432star RNG.

@param state State of the RNG to use.
*/
#define xorshift6432star_uint(state) _xorshift6432star_uint(&state)
unsigned int _xorshift6432star_uint(__global xorshift6432star_state* restrict state){
	*state ^= *state >> 12; // a
	*state ^= *state << 25; // b
	*state ^= *state >> 27; // c
	return (uint)((*state*0x2545F4914F6CDD1D)>>32);
}

/**
Seeds xorshift6432star RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void xorshift6432star_seed(__global xorshift6432star_state* state, unsigned long j){
	if(j==0){
		j++;
	}
	*state=j;
}

/**
Generates a random 64-bit unsigned integer using xorshift6432star RNG.

@param state State of the RNG to use.
*/
#define xorshift6432star_ulong(state) ((((ulong)xorshift6432star_uint(state)) << 32) | xorshift6432star_uint(state))

__kernel void xorshift6432star_init(__global xorshift6432star_state *states, __global ulong *seeds)
{
    const uint gid = get_global_id(0);
    xorshift6432star_seed(&states[gid], seeds[gid]);
}

__kernel void xorshift6432star_generate(__global xorshift6432star_state *states, __global ulong *res)
{
    const uint gid = get_global_id(0);
    res[gid] = xorshift6432star_ulong(states[gid]);
}
