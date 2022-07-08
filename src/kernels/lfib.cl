/**
@file

Implements a Multiplicative Lagged Fibbonaci generator. Returns 64-bit random numbers, but the lowest bit is always 1.

G. Marsaglia, L.-H. Tsay, Matrices and the structure of random number sequences, Linear algebra and its applications 67 (1985) 147–156.
*/


#define LFIB_LAG1 17
#define LFIB_LAG2 5

/**
State of lfib RNG.
*/
typedef struct{
	ulong s[LFIB_LAG1];
	char p1,p2;
}lfib_state;

/**
Generates a random 64-bit unsigned integer using lfib RNG.

@param state State of the RNG to use.
*/
#define lfib_ulong(state) _lfib_ulong(&state)
ulong _lfib_ulong(__global lfib_state* state){
	/*state->p1++;
	state->p1%=LFIB_LAG1;
	state->p2++;
	state->p2%=LFIB_LAG2;*/
	state->p1--;
	if(state->p1<0) state->p1=LFIB_LAG1-1;
	state->p2--;
	if(state->p2<0) state->p2=LFIB_LAG1-1;
	state->s[state->p1]*=state->s[state->p2];
	return state->s[state->p1];
}

/**
Seeds lfib RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void lfib_seed(__global lfib_state* state, ulong j){
	state->p1=LFIB_LAG1;
	state->p2=LFIB_LAG2;
	//if(get_global_id(0)==0) printf("seed %d\n",state->p1);
    for (int i = 0; i < LFIB_LAG1; i++){
		j=6906969069UL * j + 1234567UL; //LCG
		state->s[i] = j | 1; // values must be odd
	}
}

__kernel void lfib_init(__global lfib_state *states, __global ulong *seeds)
{
    const uint gid = get_global_id(0);
    lfib_seed(&states[gid], seeds[gid]);
}

__kernel void lfib_generate(__global lfib_state *states, __global ulong *res)
{
    const uint gid = get_global_id(0);
    res[gid] = lfib_ulong(states[gid]);
}
