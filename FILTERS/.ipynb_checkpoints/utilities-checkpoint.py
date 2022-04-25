# %%
import numpy as np
# %%
def strided_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(S*n,n))


# %%
def strided_app0(a, L):
    # Store the shape and strides info
    shp = a.shape
    s  = a.strides

    # Compute length of output array along the first axis
    nd0 = shp[0]-L+1

    # Setup shape and strides for use with np.lib.stride_tricks.as_strided
    # and get (n+1) dim output array
    shp_in = (nd0,L)+shp[1:]
    strd_in = (s[0],) + s
    return np.lib.stride_tricks.as_strided(a, shape=shp_in, strides=strd_in)
##



# %%
def strided_app1(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))



# %%
def strided_app2(a, L, S):  # Window len = L, Stride len/stepsize = S
    d = np.array(
        [np.lib.stride_tricks.as_strided(a[:, i], shape=(((a[:, i].size - L) // S) + 1, L),
        strides=(S * a[:, i].strides[0], a[:, i].strides[0])) for i in range(a.shape[1])])
    return d



# %%
def shift(arr, num):
    result = np.empty_like(arr)

    if num > 0:
        result[:num, :] = np.nan
        result[num:, :] = arr[:-num, :]
    elif num < 0:
        result[num:, :] = np.nan
        result[:num, :] = arr[-num:, :]
    else:
        result[:, :] = arr
    return result



# %%
def numpy_fill(arr):
    '''numpy foreward fill.'''
    mask = np.isnan(arr)
    if mask.sum() != 0:
        idx = np.where(~mask,np.arange(mask.shape[1]),0)
        np.maximum.accumulate(idx,axis=1, out=idx)
        out = arr[np.arange(idx.shape[0])[:,None], idx]
    else:
        out = arr
    return out

