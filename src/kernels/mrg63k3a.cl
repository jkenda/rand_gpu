/**
@file

Implements mrg63k3a RNG.

P. L’ecuyer, Good parameters and implementations for combined multiple recursive random number generators, Operations Research 47 (1) (1999) 159–164.
*/


#define MRG63K3A_M1    9223372036854769163
#define MRG63K3A_M2    9223372036854754679
#define MRG63K3A_A12   1754669720
#define MRG63K3A_Q12   5256471877
#define MRG63K3A_R12   251304723
#define MRG63K3A_A13N  3182104042
#define MRG63K3A_Q13   2898513661
#define MRG63K3A_R13   394451401
#define MRG63K3A_A21   31387477935
#define MRG63K3A_Q21   293855150
#define MRG63K3A_R21   143639429
#define MRG63K3A_A32N  6199136374
#define MRG63K3A_Q23   1487847900
#define MRG63K3A_R23   985240079

/**
State of mrg63k3a RNG.
*/
typedef struct{
	long s10, s11, s12, s20, s21, s22;
} mrg63k3a_state;

/**
Internal function. Advances state of mrg63k3a RNG and returns generated number.

@param state Pointer to state of the RNG to use.
*/
ulong mrg63k3a_advance(__global mrg63k3a_state* state){
	long h, p12, p13, p21, p23;
	/* Component 1 */
	h = state->s10 / MRG63K3A_Q13;
	p13 = MRG63K3A_A13N * (state->s10 - h * MRG63K3A_Q13) - h * MRG63K3A_R13;
	h = state->s11 / MRG63K3A_Q12;
	p12 = MRG63K3A_A12 * (state->s11 - h * MRG63K3A_Q12) - h * MRG63K3A_R12;
	if (p13 < 0)
		p13 += MRG63K3A_M1;
	if (p12 < 0)
		p12 += MRG63K3A_M1 - p13;
	else
		p12 -= p13;
	if (p12 < 0)
		p12 += MRG63K3A_M1;
	state->s10 = state->s11;
	state->s11 = state->s12;
	state->s12 = p12;

	/* Component 2 */
	h = state->s20 / MRG63K3A_Q23;
	p23 = MRG63K3A_A32N * (state->s20 - h * MRG63K3A_Q23) - h * MRG63K3A_R23;
	h = state->s22 / MRG63K3A_Q21;
	p21 = MRG63K3A_A21 * (state->s22 - h * MRG63K3A_Q21) - h * MRG63K3A_R21;
	if (p23 < 0)
		p23 += MRG63K3A_M2;
	if (p21 < 0)
		p21 += MRG63K3A_M2 - p23;
	else
		p21 -= p23;
	if (p21 < 0)
		p21 += MRG63K3A_M2;
	state->s20 = state->s21;
	state->s21 = state->s22;
	state->s22 = p21;

	/* Combination */
	if (p12 > p21)
		return p12 - p21;
	else
		return p12 - p21 + MRG63K3A_M1;
}

/**
Seeds mrg63k3a RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each instance of generator (thread).
*/
void mrg63k3a_seed(__global mrg63k3a_state* state, ulong j){
	state->s10 = j;
	state->s11 = j;
	state->s12 = j;
	state->s20 = j;
	state->s21 = j;
	state->s22 = j;
	if(j == 0){
		state->s10++;
		state->s21++;
	}
}

/**
Generates a random 64-bit unsigned integer using mrg63k3a RNG.

@param state State of the RNG to use.
*/
#define mrg63k3a_ulong(state) (mrg63k3a_advance(&state) << 1)//_mrg63k3a_ulong(&state)
//mrg63k3a generates only 63 random bits - MSB is always 0. We shift output, since TestU01 ignores LSB.
ulong _mrg63k3a_ulong(__global mrg63k3a_state* state){
	return mrg63k3a_advance(state) << 1;
}

__kernel void mrg63k3a_init(__global mrg63k3a_state *states, __global ulong *seeds)
{
    const uint gid = get_global_id(0);
    mrg63k3a_seed(&states[gid], seeds[gid]);
}

__kernel void mrg63k3a_generate(__global mrg63k3a_state *states, __global ulong *res)
{
    const uint gid = get_global_id(0);
    res[gid] = mrg63k3a_ulong(states[gid]);
}
