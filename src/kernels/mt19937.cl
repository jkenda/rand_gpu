/**
@file

Implements Mersenne twister generator. 

M. Matsumoto, T. Nishimura, Mersenne twister: a 623-dimensionally equidistributed uniform pseudo-random number generator, ACM Transactions on Modeling and Computer Simulation (TOMACS) 8 (1) (1998) 3–30.
*/

typedef int aligned_int __attribute__((aligned(8)));

#define MT19937_N 624
#define MT19937_M 397
#define MT19937_MATRIX_A 0x9908b0df   /* constant vector a */
#define MT19937_UPPER_MASK 0x80000000 /* most significant w-r bits */
#define MT19937_LOWER_MASK 0x7fffffff /* least significant r bits */


/**
State of MT19937 RNG.
*/
typedef struct __attribute__((aligned(16))){
	uint mt[MT19937_N]; /* the array for the state vector  */
	aligned_int mti;
} mt19937_state;

/* MAG01[x] = x * MT19937_MATRIX_A  for x=0,1 */
__constant uint MAG01[2]={0x0, MT19937_MATRIX_A};

/**
Generates a random 32-bit unsigned integer using MT19937 RNG.

@param state State of the RNG to use.
*/
#define mt19937_uint(state) _mt19937_uint(&state)
uint _mt19937_uint(__global mt19937_state* state){
    uint y;
	
	if(state->mti<MT19937_N-MT19937_M){
		y = (state->mt[state->mti]&MT19937_UPPER_MASK)|(state->mt[state->mti+1]&MT19937_LOWER_MASK);
		state->mt[state->mti] = state->mt[state->mti+MT19937_M] ^ (y >> 1) ^ MAG01[y & 0x1];
	}
	else if(state->mti<MT19937_N-1){
		y = (state->mt[state->mti]&MT19937_UPPER_MASK)|(state->mt[state->mti+1]&MT19937_LOWER_MASK);
		state->mt[state->mti] = state->mt[state->mti+(MT19937_M-MT19937_N)] ^ (y >> 1) ^ MAG01[y & 0x1];
	}
	else{
        y = (state->mt[MT19937_N-1]&MT19937_UPPER_MASK)|(state->mt[0]&MT19937_LOWER_MASK);
        state->mt[MT19937_N-1] = state->mt[MT19937_M-1] ^ (y >> 1) ^ MAG01[y & 0x1];
        state->mti = 0;
	}
    y = state->mt[state->mti++];
		
    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= (y >> 18);

    return y;
}

/**
Seeds MT19937 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void mt19937_seed(__global mt19937_state* state, uint s){
    state->mt[0]= s;
	uint mti;
    #pragma unroll
    for (mti=1; mti<MT19937_N; mti++) {
        state->mt[mti] = 1812433253 * (state->mt[mti-1] ^ (state->mt[mti-1] >> 30)) + mti;
		
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt19937[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
    }
	state->mti=mti;
}

/**
Generates a random 64-bit unsigned integer using MT19937 RNG.

@param state State of the RNG to use.
*/
#define mt19937_ulong(state) ((((ulong)mt19937_uint(state)) << 32) | mt19937_uint(state))

__kernel void mt19937_init(__global mt19937_state *states, __global uint *seeds)
{
    const uint gid = get_global_id(0);
    mt19937_seed(&states[gid], seeds[gid]);
}

__kernel void mt19937_generate(__global mt19937_state *states, __global ulong *res)
{
    const uint gid = get_global_id(0);
    res[gid] = mt19937_ulong(states[gid]);
}
