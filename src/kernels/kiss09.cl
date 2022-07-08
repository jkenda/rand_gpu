/**
@file

Implements KISS (Keep It Simple, Stupid) generator, proposed in 2009.

G. Marsaglia, 64-bit kiss rngs, https://www.thecodingforums.com/threads/64-bit-kiss-rngs.673657.
*/


//https://www.thecodingforums.com/threads/64-bit-kiss-rngs.673657/

/**
State of kiss09 RNG.
*/
typedef struct {
	ulong x,c,y,z;
} kiss09_state;

/**
Generates a random 64-bit unsigned integer using kiss09 RNG.

@param state State of the RNG to use.
*/
#define kiss09_ulong(state) (\
	/*multiply with carry*/ \
	state.c = state.x >> 6, \
	state.x += (state.x << 58) + state.c, \
	state.c += state.x < (state.x << 58) + state.c, \
	/*xorshift*/ \
	state.y ^= state.y << 13, \
	state.y ^= state.y >> 17, \
	state.y ^= state.y << 43, \
	/*linear congruential*/ \
	state.z = 6906969069UL * state.z + 1234567UL, \
	state.x + state.y + state.z \
	)

/**
Seeds kiss09 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void kiss09_seed(__global kiss09_state* state, ulong j){
	state->x = 1234567890987654321UL ^ j;
	state->c = 123456123456123456UL ^ j;
	state->y = 362436362436362436UL ^ j;
	if(state->y==0){
		state->y=1;
	}
	state->z = 1066149217761810UL ^ j;
}

__kernel void kiss09_init(__global kiss09_state *states, __global ulong *seeds)
{
    const uint gid = get_global_id(0);
    kiss09_seed(&states[gid], seeds[gid]);
}

__kernel void kiss09_generate(__global kiss09_state *states, __global ulong *res)
{
    const uint gid = get_global_id(0);
    res[gid] = kiss09_ulong(states[gid]);
}
