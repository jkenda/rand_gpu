#include "tyche_i.cl"

uint gid;
tyche_i_state state;
ulong rand;

__kernel void init(__global ulong *seed)
{
    gid = get_global_id(0);
    tyche_i_seed(&state, seed[gid]);
    rand = tyche_i_ulong(state);
}

__kernel void generate()
{
    rand = tyche_i_ulong(state);
}

__kernel void get(__global ulong *res)
{
    res[gid] = rand;
}
