/**
Modified tyche_i.cl from RandomCL (https://github.com/bstatcomp/RandomCL)
 */

/**
State of tyche_i RNG.
*/
typedef union{
	struct{
		uint a,b,c,d;
	};
	ulong res;
} tyche_i_state;

#define TYCHE_I_ROT(a,b) (((a) >> (b)) | ((a) << (32 - (b))))

/**
Generates a random 64-bit unsigned integer using tyche_i RNG.

@param state State of the RNG to use.
*/
#define tyche_i_ulong(state) (tyche_i_advance(&state), state.res)
void tyche_i_advance(__global tyche_i_state* state){
	state->b = TYCHE_I_ROT(state->b, 7) ^ state->c;
	state->c -= state->d;
	state->d = TYCHE_I_ROT(state->d, 8) ^ state->a;
	state->a -= state->b;
	state->b = TYCHE_I_ROT(state->b, 12) ^ state->c;
	state->c -= state->d;
	state->d = TYCHE_I_ROT(state->d, 16) ^ state->a;
	state->a -= state->b;
}

/**
Seeds tyche_i RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void tyche_i_seed(__global tyche_i_state* state, ulong seed){
	state->a = seed >> 32;
	state->b = seed;
	state->c = 2654435769;
	state->d = 1367130551 ^ (get_global_id(0) + get_global_size(0) * (get_global_id(1) + get_global_size(1) * get_global_id(2)));
	for(uint i=0;i<20;i++){
		tyche_i_advance(state);
	}
}

/**
Generates a random 32-bit unsigned integer using tyche_i RNG.

@param state State of the RNG to use.
*/
#define tyche_i_uint(state) ((uint)tyche_i_ulong(state))


__kernel void init(__global tyche_i_state *states, __global ulong *seed)
{
    uint gid = get_global_id(0);
    tyche_i_seed(&states[gid], seed[gid]);
}

__kernel void generate64(__global tyche_i_state *states, __global ulong *res)
{
    uint gid = get_global_id(0);
    res[gid] = tyche_i_ulong(states[gid]);
}

__kernel void generate32(__global tyche_i_state *states, __global uint *res)
{
    uint gid = get_global_id(0);
    res[gid] = tyche_i_uint(states[gid]);
}
