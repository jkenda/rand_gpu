/**
@file

Implements msws (Middle Square Weyl Sequence) RNG.

B. Widynski, Middle square weyl sequence rng, arXiv preprint arXiv:1704.00358. https://arxiv.org/abs/1704.00358
*/


/**
State of msws RNG.
*/
typedef struct{
	union{
		ulong x;
		uint2 x2;
	};
	ulong w;
}msws_state;

/**
Generates a random 64-bit unsigned integer using msws RNG.

This is alternative, macro implementation of msws RNG.

@param state State of the RNG to use.
*/
#define msws_macro_ulong(state) (\
	state.x *= state.x, \
	state.x += (state.w += 0xb5ad4eceda1ce2a9), \
	state.x = (state.x>>32) | (state.x<<32) \
	)

/**
Generates a random 64-bit unsigned integer using msws RNG.

@param state State of the RNG to use.
*/
#define msws_ulong(state) _msws_ulong(&state)
ulong _msws_ulong(__global msws_state* state){
	state->x *= state->x;
	state->x += (state->w += 0xb5ad4eceda1ce2a9);
	return state->x = (state->x>>32) | (state->x<<32);
}

/**
Seeds msws RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void msws_seed(__global msws_state* state, ulong j){
	state->x = j;
	state->w = j;
}

__kernel void msws_init(__global msws_state *states, __global ulong *seeds)
{
    const uint gid = get_global_id(0);
    msws_seed(&states[gid], seeds[gid]);
}

__kernel void msws_generate(__global msws_state *states, __global ulong *res)
{
    const uint gid = get_global_id(0);
    res[gid] = msws_ulong(states[gid]);
}
