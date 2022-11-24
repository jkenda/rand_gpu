/**
@file

Implements a 512-bit tyche (Well-Equidistributed Long-period Linear) RNG.

S. Neves, F. Araujo, Fast and small nonlinear pseudorandom number generators for computer simulation, in: International Conference on Parallel Processing and Applied Mathematics, Springer, 2011, pp. 92–101.
*/


/**
State of tyche RNG.
*/
typedef union{
	struct{
		uint a,b,c,d;
	};
	ulong res;
} tyche_state;

#define TYCHE_ROT(a,b) (((a) << (b)) | ((a) >> (32 - (b))))

/**
Generates a random 64-bit unsigned integer using tyche RNG.

@param state State of the RNG to use.
*/
#define tyche_ulong(state) (tyche_advance(&state), state.res)
void tyche_advance(__global tyche_state* state){
	state->a += state->b;
	state->d = TYCHE_ROT(state->d ^ state->a, 16);
	state->c += state->d;
	state->b = TYCHE_ROT(state->b ^ state->c, 12);
	state->a += state->b;
	state->d = TYCHE_ROT(state->d ^ state->a, 8);
	state->c += state->d;
	state->b = TYCHE_ROT(state->b ^ state->c, 7);
}

/**
Seeds tyche RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void tyche_seed(__global tyche_state* state, ulong seed){
	state->a = seed >> 32;
	state->b = seed;
	state->c = 2654435769;
	state->d = 1367130551 ^ get_global_id(0);
	#pragma unroll
	for(uint i=0;i<20;i++){
		tyche_advance(state);
	}
}

__kernel void tyche_init(__global tyche_state *states, ulong seed)
{
    const uint gid = get_global_id(0);
    tyche_seed(&states[gid], seed);
}

__kernel void tyche_generate(__global tyche_state *states, __global ulong *res)
{
    const uint gid = get_global_id(0);
    res[gid] = tyche_ulong(states[gid]);
}
