import numba as nb

@nb.njit(cache=True)
def allocate_fs_block(fs_usage_uv, start_fs, end_fs, Bool):
    for fs in range(start_fs, end_fs + 1):
        fs_usage_uv[fs] = Bool