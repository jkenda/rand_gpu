/**
@file

Implements a ran2 RNG.

W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery, Numerical recipes in c: The art of scientific computing (; cambridge (1992).
*/


#define   IM1 2147483563
#define   IM2 2147483399
#define   AM (1.0/IM1)
#define   IMM1  (IM1-1)
#define   IA1 40014
#define   IA2 40692
#define   IQ1 53668
#define   IQ2 52774
#define   IR1 12211 
#define   IR2 3791
#define   NTAB 32
#define   NDIV (1+IMM1/NTAB)
#define   EPS 1.2e-7
#define   RNMX (1.0-EPS)

/**
State of ran2 RNG.
*/
typedef struct{
	int idum;
	int idum2;
	int iy;
	int iv[NTAB];
} ran2_state;

/**
Generates a random 32-bit unsigned integer using ran2 RNG. The lowest bit is always 0.

@param state State of the RNG to use.
*/
uint ran2_uint(__global ran2_state* state){
	
	int k = state->idum / IQ1;
	state->idum = IA1 * (state->idum - k*IQ1) - k*IR1;
	if(state->idum < 0){
		state->idum += IM1;
	}
	
	k = state->idum2 / IQ2;
	state->idum2 = IA2 * (state->idum2 - k*IQ2) - k*IR2;
	if(state->idum2 < 0){
		state->idum2 += IM2;
	}
	
	short j = state->iy / NDIV;
	state->iy = state->iv[j] - state->idum2;
	state->iv[j] = state->idum;
	if(state->iy < 1){
		state->iy += IMM1;
	}
	return state->iy;
	/*float temp = AM * state->iy;
	if(temp > RNMX){
		return RNMX;
	}
	else {
		return temp;
	}*/
}

/**
Seeds ran2 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void ran2_seed(__global ran2_state* state, ulong seed){
	if(seed == 0){
		seed = 1;
	}
	state->idum = seed;
	state->idum2 = seed>>32;
	for(int j = NTAB + 7; j >= 0; j--){
		short k = state->idum / IQ1;
		state->idum = IA1 * (state->idum - k*IQ1) - k*IR1;
		if(state->idum < 0){
			state->idum += IM1;
		}
		if(j < NTAB){
			state->iv[j] = state->idum;
		}
	}
	state->iy = state->iv[0];
}

/**
Generates a random 64-bit unsigned integer using ran2 RNG.

@param state State of the RNG to use.
*/
#define ran2_ulong(state) ( (((ulong) ran2_uint(&state)) << 33) | (((ulong) ran2_uint(&state)) << 2) | ran2_uint(&state) & 0b11 )

__kernel void ran2_init(__global ran2_state *states, __global ulong *seeds)
{
    const uint gid = get_global_id(0);
    ran2_seed(&states[gid], seeds[gid]);
}

__kernel void ran2_generate(__global ran2_state *states, __global ulong *res)
{
    const uint gid = get_global_id(0);
    res[gid] = ran2_ulong(states[gid]);
}
