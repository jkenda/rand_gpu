/**
Modified tyche_i.cl from RandomCL (https://github.com/bstatcomp/RandomCL)
 */

typedef union{
	struct{
		uint a,b,c,d;
	};
	ulong res;
} tyche_i_state;

#define TYCHE_I_ROT(a,b) (((a) >> (b)) | ((a) << (32 - (b))))

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

void tyche_i_seed(__global tyche_i_state* state, ulong seed){
	state->a = seed >> 32;
	state->b = seed;
	state->c = 2654435769;
	state->d = 1367130551 ^ (get_global_id(0) + get_global_size(0) * (get_global_id(1) + get_global_size(1) * get_global_id(2)));
	for(uint i=0;i<20;i++){
		tyche_i_advance(state);
	}
}


__kernel void init(__global tyche_i_state *states, ulong seed)
{
    const uint gid = get_global_id(0);
    tyche_i_seed(&states[gid], seed);
}

__kernel void generate(__global tyche_i_state *states, __global ulong *res)
{
    const uint gid = get_global_id(0);
    res[gid] = tyche_i_ulong(states[gid]);
}
