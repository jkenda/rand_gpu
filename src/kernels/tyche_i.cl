/**
@file

Implements tyche-i RNG.

S. Neves, F. Araujo, Fast and small nonlinear pseudorandom number generators for computer simulation, in: International Conference on Parallel Processing and Applied Mathematics, Springer, 2011, pp. 92–101.
*/


/**
State of tyche_i RNG.
*/
typedef union{
    uint4 u;
	ulong res;
} tyche_i_state;

#define TYCHE_I_ROT(a,b) (((a) >> (b)) | ((a) << (32 - (b))))

/**
Generates a random 64-bit unsigned integer using tyche_i RNG.

@param state State of the RNG to use.
*/
#define tyche_i_ulong(state) (tyche_i_advance(&state), state.res)
void tyche_i_advance(__global tyche_i_state* state){
	state->u.y = TYCHE_I_ROT(state->u.y, 7) ^ state->u.z;
	state->u.z -= state->u.w;
	state->u.w = TYCHE_I_ROT(state->u.w, 8) ^ state->u.x;
	state->u.x -= state->u.y;
	state->u.y = TYCHE_I_ROT(state->u.y, 12) ^ state->u.z;
	state->u.z -= state->u.w;
	state->u.w = TYCHE_I_ROT(state->u.w, 16) ^ state->u.x;
	state->u.x -= state->u.y;
}

/**
Seeds tyche_i RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void tyche_i_seed(__global tyche_i_state* state, ulong seed){
	state->u.x = seed >> 32;
	state->u.y = seed;
	state->u.z = 2654435769;
	state->u.w = 1367130551 ^ (get_global_id(0) + get_global_size(0) * (get_global_id(1) + get_global_size(1) * get_global_id(2)));
	#pragma unroll
	for(uint i=0;i<20;i++){
		tyche_i_advance(state);
	}
}

__kernel void tyche_i_init(__global tyche_i_state *states, ulong seed)
{
    const uint gid = get_global_id(0);
    tyche_i_seed(&states[gid], seed);
}

__kernel void tyche_i_generate(__global tyche_i_state *states, __global ulong *res)
{
    const uint gid = get_global_id(0);
    res[gid] = tyche_i_ulong(states[gid]);
}
