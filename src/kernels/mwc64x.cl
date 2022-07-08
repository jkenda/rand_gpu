/**
@file

Implements a 64-bit Multiply With Carry generator that returns 32-bit numbers that are xor of lower and upper 32-bit numbers.

http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
*/


/**
State of mwc64x RNG.
*/
typedef union {
	ulong xc;
	struct{ 
		uint x;
		uint c;
	};
} mwc64x_state;

/**
Generates a random 32-bit unsigned integer using mwc64x RNG.

@param state State of the RNG to use.
*/
#define mwc64x_uint(state) _mwc64x_uint(&state)
uint _mwc64x_uint(__global mwc64x_state *s)
{
	uint res = s->x ^ s->c;
	uint X = s->x;
	uint C = s->c;
	
	uint Xn=4294883355U*X+C;
	uint carry=(uint)(Xn<C);
	uint Cn=mad_hi(4294883355U,X,carry); 
	
	s->x=Xn;
	s->c=Cn;
	return res;
}

/**
Seeds mwc64x RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void mwc64x_seed(__global mwc64x_state* state, unsigned long j){
	state->xc=j;
}

/**
Generates a random 64-bit unsigned integer using mwc64x RNG.

@param state State of the RNG to use.
*/
#define mwc64x_ulong(state) ((((ulong)mwc64x_uint(state)) << 32) | mwc64x_uint(state))

__kernel void mwc64x_init(__global mwc64x_state *states, __global ulong *seeds)
{
    const uint gid = get_global_id(0);
    mwc64x_seed(&states[gid], seeds[gid]);
}

__kernel void mwc64x_generate(__global mwc64x_state *states, __global ulong *res)
{
    const uint gid = get_global_id(0);
    res[gid] = mwc64x_ulong(states[gid]);
}
