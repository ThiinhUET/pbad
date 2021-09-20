# Cython implementation of the expensive subroutines

cimport cython
import numpy as np
cimport numpy as np

from libc.math cimport pow, fabs, sqrt

# types to be used
ctypedef np.npy_float32 DTYPE_t
ctypedef np.npy_float64 DOUBLE_t
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer


################################################################################
# JACCARD SIMILARITY
################################################################################

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef compute_pattern_cover(np.ndarray encoded_patterns, list encoded_data, str pattern_type, DOUBLE_t jaccard_threshold):
    # compute the cover of each pattern
    cdef int N = len(encoded_patterns)
    cdef int M = len(encoded_data)

    cdef np.ndarray[DOUBLE_t, ndim=2] covers = np.zeros((N, M), dtype=np.float64)
    cdef int i, j
    if pattern_type == 'itemset':
        for j in range(N):
            for i in range(M):
                covers[j, i] = _compute_cover_itemset(encoded_patterns[j], encoded_data[i])
    else:
        for j in range(N):
            for i in range(M):
                covers[j, i] = _compute_cover_sequential_pattern(encoded_patterns[j], encoded_data[i])

    return covers


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef remove_overlapping_jaccard(np.ndarray encoded_patterns, list encoded_data, str pattern_type, DOUBLE_t jaccard_threshold):
    # compute the cover of each pattern
    cdef int N = len(encoded_patterns)
    cdef int M = len(encoded_data)

    cdef np.ndarray[DOUBLE_t, ndim=2] covers = np.zeros((N, M), dtype=np.float64)
    cdef int i, j
    if pattern_type == 'itemset':
        for j in range(N):
            for i in range(M):
                covers[j, i] = _compute_cover_itemset(encoded_patterns[j], encoded_data[i])
    else:
        for j in range(N):
            for i in range(M):
                covers[j, i] = _compute_cover_sequential_pattern(encoded_patterns[j], encoded_data[i])

    # remove overlapping with jaccard (every pair of patterns)
    cdef np.ndarray keep_ids = np.zeros(N, dtype=int)
    cdef int ii, jj
    cdef int intersect, c1, c2
    cdef DOUBLE_t js
    cdef int keep
    for ii in range(N):
        if ii % 200 == 0:
            print('\t{}/{}'.format(ii, N))
        keep = 1
        for jj in range(ii+1, N):
            intersect = np.multiply(covers[ii, :], covers[jj, :]).sum()
            c1 = covers[ii, :].sum()
            c2 = covers[jj, :].sum()
            js = intersect / (c1 + c2 - intersect)
            if js > jaccard_threshold:
                keep = 0
                break
        if keep == 1:
            keep_ids[ii] = 1

    return keep_ids


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef DOUBLE_t _compute_cover_itemset(np.ndarray[long, ndim=1] pattern, list datap):
    cdef long v
    for v in pattern:
        if not v in datap:
            return 0.0
    return 1.0


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef DOUBLE_t _compute_cover_sequential_pattern(np.ndarray[long, ndim=1] pattern, list datap):
    cdef long v
    for v in pattern:
        if v in datap:
            idx = datap.index(v)+1
            datap = datap[idx:]
        else:
            return 0.0
    return 1.0


################################################################################
# MAKING PATTERN BASED FEATURES
################################################################################

cpdef make_pattern_based_features_cython(np.ndarray[DOUBLE_t, ndim=2] data, np.ndarray patterns, int n, int npa, double par_lambda, int formula):
    cdef np.ndarray[DOUBLE_t, ndim=2] features = np.zeros((n, npa), dtype=np.float64)
    cdef int i, j
    if formula == 1:
        for i in range(n):
            for j in range(npa):
                features[i, j] = _compute_distance_weighted_similarity_formula1(data[i, :], patterns[j], par_lambda)
    else:
        for i in range(n):
            for j in range(npa):
                features[i, j] = _compute_distance_weighted_similarity_formula2(data[i, :], patterns[j], par_lambda)
    return features


cpdef make_pattern_based_features_cython_event_logs(list logs, np.ndarray patterns, int n, int npa):
    cdef np.ndarray[DOUBLE_t, ndim=2] features = np.zeros((n, npa), dtype=np.float64)
    cdef int i, j
    for i in range(n):
        for j in range(npa):
            features[i, j] = _compute_exact_match_with_pattern(logs[i], patterns[j])
    return features


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef DOUBLE_t _compute_exact_match_with_pattern(list row, np.ndarray[long, ndim=1] pattern):
    # if length of the row < pattern: no match (cover)
    cdef int c = len(pattern)
    cdef int r = len(row)
    if r < c:
        return 0.0

    # fast implementation of distance computation
    cdef long v
    for v in pattern:
        if v in row:
            idx = row.index(v)+1
            row = row[idx:]
        else:
            return 0.0
    return 1.0


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef DOUBLE_t _compute_distance_weighted_similarity_formula1(np.ndarray[DOUBLE_t, ndim=1] row, np.ndarray[DOUBLE_t, ndim=1] pattern, double par_lambda):
    # fast implementation of distance computation
    cdef int c = pattern.size
    cdef int r = row.size
    cdef DOUBLE_t simil

    # the row has < items than the pattern

    # pattern with a single item
    cdef int i
    cdef DOUBLE_t d = 999999
    cdef DOUBLE_t d_new
    if c == 1:
        for i in range(r):
            d_new = sqrt(pow(fabs(row[i] - pattern[0]), par_lambda))
            if d_new < d:
                d = d_new
        simil = max(0.0, 1.0 - d) # c = 1
        return simil

    # construct matrix
    cdef int j
    cdef int w = r - c + 1
    cdef np.ndarray[DOUBLE_t, ndim=2] matrix = np.zeros((r+1, c+1), dtype=np.float64)
    for i in range(r+1):
        for j in range(c+1):
            if j > 0:
                matrix[i, j] = 999999

    # fill the matrix
    cdef int ii, jj
    cdef DOUBLE_t best_d
    for i in range(c):
        ii = i + 1
        for j in range(i, i + w):
            jj = j + 1
            d_new = pow(fabs(pattern[i] - row[i]), par_lambda)
            best_d = min(matrix[j, ii], d_new + matrix[j, i])
            matrix[jj, ii] = best_d

    # last element is the distance + sqrt
    cdef DOUBLE_t dist = sqrt(matrix[r, c])
    simil = max(0.0, 1.0 - (dist / c))
    return simil

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef DOUBLE_t _compute_distance_weighted_similarity_formula2(np.ndarray[DOUBLE_t, ndim=1] row, np.ndarray[DOUBLE_t, ndim=1] pattern, double par_lambda):
    # fast implementation of distance computation
    cdef int c = pattern.size
    cdef int r = row.size
    cdef DOUBLE_t simil

    # pattern with a single item
    cdef int i
    cdef DOUBLE_t d = 999999
    cdef DOUBLE_t d_new
    if c == 1:
        for i in range(r):
            d_new = pow(fabs(row[i] - pattern[0]), 1.0 / par_lambda)
            if d_new < d:
                d = d_new
        simil = max(0.0, 1.0 - d) # c = 1
        return simil

    # construct matrix
    cdef int j
    cdef int w = r - c + 1
    cdef np.ndarray[DOUBLE_t, ndim=2] matrix = np.zeros((r+1, c+1), dtype=np.float64)
    for i in range(r+1):
        for j in range(c+1):
            if j > 0:
                matrix[i, j] = 999999

    # fill the matrix
    cdef int ii, jj
    cdef DOUBLE_t best_d
    for i in range(c):
        ii = i + 1
        for j in range(i, i + w):
            jj = j + 1
            d_new = pow(fabs(pattern[i] - row[i]), 1.0 / par_lambda)
            best_d = min(matrix[j, ii], d_new + matrix[j, i])
            matrix[jj, ii] = best_d

    # last element is the distance + sqrt
    cdef DOUBLE_t dist = matrix[r, c]
    simil = max(0.0, 1.0 - (dist / c))
    return simil


################################################################################
# COMPUTING THE FPOF SCORE
################################################################################

@cython.boundscheck(False)
cpdef compute_fpof_score(np.ndarray[DOUBLE_t, ndim=2] data, np.ndarray patterns, np.ndarray supports):
    cdef int n = len(data)
    cdef int npa = len(patterns)
    cdef np.ndarray[DOUBLE_t, ndim=1] fpof_score = np.zeros(n, dtype=np.float64)

    cdef int i, j
    cdef int contained
    cdef DOUBLE_t fpof
    for i in range(n):
        fpof = 0.0
        w = data[i, :]
        for j in range(npa):
            fp = patterns[j]
            contained = _compute_contains_pattern(w, fp)
            if contained == 1:
                fpof = fpof + supports[j]
        fpof_score[i] = 1.0 - fpof / npa

    return fpof_score

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef int _compute_contains_pattern(np.ndarray w, np.ndarray fp):
    cdef int contained = 1
    cdef int i
    cdef int n = len(fp)
    for i in range(n):
        if not(fp[i] in w):
            contained = 0
            break
    return contained
