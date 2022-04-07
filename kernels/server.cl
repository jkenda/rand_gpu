#include "tyche_i.cl"

__kernel void init(__global tyche_i_state *states, __global ulong *seed)
{
    uint gid = get_global_id(0);
    tyche_i_state state;
    tyche_i_seed(&state, seed[gid]);
    states[gid] = state;
}

__kernel void generate(__global tyche_i_state *states, __global ulong *res)
{
    uint gid = get_global_id(0);
    tyche_i_state state = states[gid];
    res[gid] = tyche_i_ulong(state);
    states[gid] = state;
}
