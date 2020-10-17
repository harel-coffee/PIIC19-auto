#### FROM https://github.com/cmackenziek/tsfl

import numpy as np
import pandas as pd
from random import randint, random
from sklearn.decomposition import SparseCoder


def encode_kmeans_triangle(df, km, split=False, alpha=1):
    df = pd.DataFrame(km.transform(df)) # Transformado al espacio de cluster distance
    mean = df.mean(axis=1) # la media de cada columna
    df = df.div(-1)
    df = df.add(alpha*mean, axis=0) # Dejar todos los valores en Mu - X
    if split:
        df1 = df.apply(lambda x: np.maximum(0,x)) # Dejar todos los negativos en 0
        df2 = df.apply(lambda x: np.maximum(0,-x)) # Dejar todos los positivos en 0
        df = df1.merge(df2, left_index=True, right_index=True)
    else:
        df = df.apply(lambda x: np.maximum(0,x)) # Dejar todos los negativos en 0
    return df


def encode_kmeans_triangleok(df, km, split=False, alpha=1):
    df = pd.DataFrame(km.transform(df)) # Transformado al espacio de cluster distance
    mean = df.mean(axis=0) # la media de cada fila
    df = df.div(-1)
    df = df.add(alpha*mean, axis=1) # Dejar todos los valores en Mu - X
    if split:
        df1 = df.apply(lambda x: np.maximum(0,x)) # Dejar todos los negativos en 0
        df2 = df.apply(lambda x: np.maximum(0,-x)) # Dejar todos los positivos en 0
        df = df1.merge(df2, left_index=True, right_index=True)
    else:
        df = df.apply(lambda x: np.maximum(0,x)) # Dejar todos los negativos en 0
    return df


def encode_kmeans_hard(df, km):
    df = pd.DataFrame(km.transform(df)) # Transformado al espacio de cluster distance
    df = df.apply(lambda x: x == min(x), axis=1) # True solo donde en el indice del cluster mas cercano
    df = df.applymap(int) # Cambiar todo a int
    return df


def encode_kmeans_soft_threshold(df, km, alpha=0.5):
    df = pd.DataFrame(km.transform(df)) # Transformado al espacio de cluster distance
    df = df.apply(lambda x: np.maximum(0, x - alpha)) # Poner el 0 los que quedan negativos
    return df


def encode_kmeans_sparsecode(df, km, algo='lasso_cd', alpha=1, split=False):
    centroids = km.cluster_centers_
    D = [centroids[i]/np.linalg.norm(centroids[i]) for i in range(len(centroids))]
    D = np.array(D)
    sc = SparseCoder(D, transform_algorithm=algo, transform_alpha=alpha, split_sign=split)
    return pd.DataFrame(sc.transform(df))


def encode_lightcurve(df, km, method='triangle', alpha=1, split=False):
    if method == 'triangle':
        return encode_kmeans_triangle(df, km, split=split, alpha=alpha)
    if method == 'triangleok':
        return encode_kmeans_triangleok(df, km, split=split, alpha=alpha)
    if method == 'sparse':
        return encode_kmeans_sparsecode(df, km, alpha=alpha, split=split)
    if method == 'threshold':
        return encode_kmeans_soft_threshold(df, km, alpha)


def encode_lightcurve_twed(cluster_lcs, cluster_times, patches_lcs,
                           patches_times, twed_func, lam=.5, nu=1e-5, alpha=1,
                           split=True):
    num_patches = len(patches_lcs)
    num_centroids = len(cluster_lcs)
    D = np.zeros((num_patches, num_centroids))

    for i in xrange(num_patches):
        for j in xrange(num_centroids):
            D[i, j] = twed_func(patches_lcs[i], patches_times[i],
                                cluster_lcs[j], cluster_times[j],
                                lam=lam, nu=nu)

    # Transformado al espacio de cluster distance
    df = pd.DataFrame(D)
    mean = df.mean(axis=1)  # la media de cada columna
    df = df.div(-1)
    df = df.add(alpha*mean, axis=0)  # Dejar todos los valores en Mu - X
    if split:
        # Dejar todos los negativos en 0
        df1 = df.apply(lambda x: np.maximum(0, x))
        # Dejar todos los positivos en 0
        df2 = df.apply(lambda x: np.maximum(0, -x))
        df = df1.merge(df2, left_index=True, right_index=True)
    else:
        # Dejar todos los negativos en 0
        df = df.apply(lambda x: np.maximum(0, x))
    return df


def max_pool(df, num_pool):
    N = len(df)
    pooled_datum = []
    for q in range(num_pool):
        pooled_datum = pooled_datum + df.iloc[q*(N/num_pool):(q+1)*(N/num_pool) - 1,:].max(axis=0).tolist()
    return pooled_datum


def mean_pool(df, num_pool):
    N = len(df)
    pooled_datum = []
    for q in range(num_pool):
        pooled_datum = pooled_datum + df.iloc[q*(N/num_pool):(q+1)*(N/num_pool) - 1,:].mean(axis=0).tolist()
    return pooled_datum


def median_pool(df, num_pool):
    N = len(df)
    pooled_datum = []
    for q in range(num_pool):
        pooled_datum = pooled_datum + df.iloc[q*(N/num_pool):(q+1)*(N/num_pool) - 1,:].median(axis=0).tolist()
    return pooled_datum


def sum_pool(df, num_pool):
    N = len(df)
    pooled_datum = []
    for q in range(num_pool):
        pooled_datum = pooled_datum + df.iloc[q*(N/num_pool):(q+1)*(N/num_pool) - 1,:].sum(axis=0).tolist()
    return pooled_datum


def pool_lightcurve(df, num_pool=4, method='mean'):
    if method == 'max':
        return max_pool(df, num_pool)
    if method == 'mean':
        return mean_pool(df, num_pool)
    if method == 'median':
        return median_pool(df, num_pool)
    if method == 'sum':
        return sum_pool(df, num_pool)


from numba import jit, float64

max_float = np.finfo(np.float).max

@jit(float64[:, :](float64[:, :], float64[:], float64[:], float64[:],
     float64[:], float64, float64), nopython=True)
def pairwise_tweds(TWED, A, A_times, B, B_times, lam=0.5, nu=1e-5):
    for i in range(1, len(A)):
        for j in range(1, len(B)):
            TWED[i, j] = min(
                # insertion
                (TWED[i - 1, j] + abs(A[i - 1] - A[i]) +
                 nu*(A_times[i] - A_times[i - 1]) + lam),
                # deletion
                (TWED[i, j - 1] + abs(B[j - 1] - B[j]) +
                 nu*(B_times[j] - B_times[j - 1]) + lam),
                # match
                (TWED[i - 1, j - 1] + abs(A[i] - B[j]) +
                 nu*(A_times[i] - B_times[j]) +
                 abs(A[i - 1] - B[j - 1]) +
                 nu*(A_times[i - 1] - B_times[j - 1]))
            )
    return TWED


def twed(A, A_times, B, B_times, lam=0.5, nu=1e-5):
    n, m = len(A), len(B)

    A, A_times = np.append(0.0, A), np.append(0.0, A_times)
    B, B_times = np.append(0.0, B), np.append(0.0, B_times)

    TWED = np.zeros((n + 1, m + 1))
    TWED[:, 0] = np.finfo(np.float).max
    TWED[0, :] = np.finfo(np.float).max
    TWED[0, 0] = 0.0

    TWED = pairwise_tweds(TWED, A, A_times, B, B_times, lam=lam, nu=nu)
    return TWED[n, m]


def encode_lightcurve_twed(cluster_lcs, cluster_times, patches_lcs,
                           patches_times, twed_lambda, twed_nu, alpha=1, split=True):
    num_patches = len(patches_lcs)
    num_centroids = len(cluster_lcs)
    D = np.zeros((num_patches, num_centroids))

    for i in range(num_patches):
        for j in range(num_centroids):
            A, A_times = patches_lcs[i], patches_times[i]
            B, B_times = cluster_lcs[j], cluster_times[j]
            D[i, j] = twed(A, A_times, B, B_times, lam=twed_lambda,
                           nu=twed_nu)

    # Transformado al espacio de cluster distance
    df = pd.DataFrame(D)
    mean = df.mean(axis=1)  # la media de cada columna
    df = df.div(-1)
    df = df.add(alpha*mean, axis=0)  # Dejar todos los valores en Mu - X
    if split:
        # Dejar todos los negativos en 0
        df1 = df.apply(lambda x: np.maximum(0, x))
        # Dejar todos los positivos en 0
        df2 = df.apply(lambda x: np.maximum(0, -x))
        df = df1.merge(df2, left_index=True, right_index=True)
    else:
        # Dejar todos los negativos en 0
        df = df.apply(lambda x: np.maximum(0, x))
    return df


def complexity_coeff(lc_a, times_a, lc_b, times_b):
    complexity_1 = np.sum(np.sqrt(np.power(np.diff(lc_a), 2)) +
                          np.sqrt(np.power(np.diff(times_a), 2)))
    complexity_2 = np.sum(np.sqrt(np.power(np.diff(lc_b), 2)) +
                          np.sqrt(np.power(np.diff(times_b), 2)))
    return max(complexity_1, complexity_2)/min(complexity_1, complexity_2)


def get_coords(N, num_twed):
    """
    This function returns the coordinates corresponding to the num_twed.
    TODO: explain well
    """
    i, j = 0, 0
    sub = N
    while num_twed > sub:
        num_twed -= sub
        sub -= 1
        i += 1
    return (i, i + num_twed - 1)


def pairwise_matrix_tweds(lcs, times, part, num_parts,lam=0.5, nu=1e-5):
    """
    For parallelization in many jobs, we calculate how many pairwise distances
    we need to calculate, and then figure out the border indexes of the ones we
    need to do in the current job
    """
    N = len(lcs)
    num_tweds = N*(N + 1)/2
    tweds_per_part = num_tweds/num_parts
    begin_index = int((part - 1)*tweds_per_part)
    if part == num_parts:
        end_index = num_tweds
    else:
        end_index = tweds_per_part*part
    end_index = int(end_index)
    print("N: {0}".format(N))
    print("num_tweds: {0}".format(num_tweds))
    print("tweds_per_part: {0}".format(tweds_per_part))
    print("begin_index: {0}".format(begin_index))
    print("end_index: {0}".format(end_index))
    print("part: {0}".format(part))
    print("num_parts: {0}".format(num_parts))
    D = np.zeros((N, N))
    k = 0
    for current_twed in range(begin_index, end_index):
        i, j = get_coords(N, current_twed + 1)
        twed_val = twed(lcs[i], times[i], lcs[j], times[j], lam=lam, nu=nu)
        complex_coeff = complexity_coeff(lcs[i], times[i], lcs[j], times[j])
        final_val = twed_val*complex_coeff
        D[i, j] = final_val
        D[j, i] = final_val
        k += 1
    return D