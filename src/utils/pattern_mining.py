"""
Mine itemsets or sequential patterns


Remark: We could also encode each item as combination of hour + value

relative_minsup = 0.001 #-> For Store 1
            #interpretation: value-hour occurs at least once every week
            #e.g. number of windows (or hours): 24408 / 24 =  1017: 1017 = expected support if it occurs every time
"""

import sys
import sys, copy
import itertools
import numpy as np
from datetime import datetime

try:
    from utils.pattern_mining_spmf import *
except ImportError:
    raise Exception('ERROR: could not load from `pattern_mining_spmf` in `pattern_mining`')

# import cython functions
try:
    from cython_utils import cpatternm as cpm
except ImportError:
    from utils.cython_utils import cpatternm as cpm
    # print('Loading CPATTERNM in PATTERN_MINING from TOP directory.')


################################################################################
# MAIN FUNCTIONS
################################################################################

def mine_non_redundant_itemsets(data, relative_minsup, jaccard_threshold=0.9, pruning='closed', include_support=False, return_encoded=False):
    """ Find non-redundant frequent itemsets

    Non-redundant itemsers == itemsets that are frequent,
    closed and do not overlap the same transactions.

    parameters
    ----------
    data : numpy array
        Each row is one window of the time series.
    relative_minsup : float
        Relative minimal support for a frequent pattern.
    jaccard_threshold : float
        Jaccard threshold to remove overlapping patterns.
    pruning : str
        'closed' or 'maximal' pruning.
    include_support : bool
        Return the support of the patterns.
    return_encoded : bool
        Return an encoded version of the data and patterns.

    returns
    -------
    patterns : numpy array
        Each element is a frequent pattern.
    supports : numpy array
        Support of each pattern.
    """

    temp_dir = make_temp_dir()

    # reduce precision
    data = data.copy()
    try:
        data = np.around(data, 10)
    except:
        pass

    # encode the data
    encode_dct, encoded_data = data_encode_as_int(data)
    decode_dct = {v: k for k, v in encode_dct.items()}

    # mine the closed or maximal patterns (encoded)
    if pruning == 'closed':
        encoded_patterns, supports = mine_closed_itemsets(encoded_data, relative_minsup, temp_dir)
    elif pruning == 'maximal':
        encoded_patterns, supports = mine_maximal_itemsets(encoded_data, relative_minsup, temp_dir)
    else:
        print('ERROR: Not a valid pruning option for mining the patterns')
        return None
    # make sure the patterns are sorted by support!
    ix_sort = np.argsort(supports)[::-1]
    encoded_patterns = encoded_patterns[ix_sort]
    supports = supports[ix_sort]
    print('DEBUG: Found #{} patterns'.format(len(encoded_patterns)))

    # remove the overlapping patterns with Jaccard
    if jaccard_threshold > 0.0 and jaccard_threshold < 1.0:
        list_encoded_data = encoded_data.tolist()
        if not(isinstance(list_encoded_data[0], list)):
            list_encoded_data = [e.tolist() for e in list_encoded_data]
        keep_ids = _remove_overlapping_patterns_jaccard(encoded_patterns, list_encoded_data, 'itemset', jaccard_threshold)
        encoded_patterns = encoded_patterns[keep_ids]
        supports = supports[keep_ids]
        print('DEBUG: # Jaccard thresholded patterns: {}'.format(len(keep_ids)))

    # decode the patterns: encoded_patters = array of arrays
    patterns = patterns_decode_from_int(encoded_patterns, decode_dct)
    npa = len(patterns)

    # report patterns
    print('DEBUG: most frequent patterns: {}'.format([(patterns[i], supports[i]) for i in range(min(5, npa))]))
    print('DEBUG: least frequent patterns: {}'.format([(patterns[i], supports[i]) for i in range(max(0, npa-5), npa)]))

    remove_temp_dir(temp_dir)
    if return_encoded:
        return encoded_patterns, encoded_data
    if include_support:
        return patterns, supports
    return patterns


def mine_non_redundant_sequential_patterns(data, relative_minsup, jaccard_threshold=0.9, pruning='closed', include_support=False, return_encoded=False):
    """ Find non-redundant frequent sequential patterns

    Non-redundant itemsers == itemsets that are frequent,
    closed and do not overlap the same transactions.

    parameters
    ----------
    data : numpy array
        Each row is one window of the time series.
    relative_minsup : float
        Relative minimal support for a frequent pattern.
    jaccard_threshold : float
        Jaccard threshold to remove overlapping patterns.
    pruning : str
        'closed' or 'maximal' pruning.
    include_support : bool
        Return the support of the patterns.
    return_encoded : bool
        Return an encoded version of the data and patterns.

    returns
    -------
    patterns : numpy array
        Each element is a frequent pattern.
    supports : numpy array
        Support of each pattern.
    """

    temp_dir = make_temp_dir()

    # reduce precision
    data = data.copy()
    try:
        data = np.around(data, 10)
    except:
        pass

    # encode the data: data are not sorted!
    encode_dct, encoded_data = data_encode_as_int(data)
    decode_dct = {v: k for k, v in encode_dct.items()}

    # mine the closed or maximal patterns (encoded)
    if pruning == 'closed':
        encoded_patterns, supports = mine_closed_sequential_patterns(encoded_data, relative_minsup, temp_dir)
    elif pruning == 'maximal':
        encoded_patterns, supports = mine_maximal_sequential_patterns(encoded_data, relative_minsup, temp_dir)
    else:
        print('ERROR: Not a valid pruning option for mining the patterns')
        return None
    # make sure the patterns are sorted by support!
    # the patterns are internally sorted!
    # all patterns have supports > 0 --> impossible that they have no cover
    ix_sort = np.argsort(supports)[::-1]
    encoded_patterns = encoded_patterns[ix_sort]
    supports = supports[ix_sort]
    print('DEBUG: Found #{} patterns'.format(len(encoded_patterns)))

    # remove the overlapping patterns with Jaccard
    if jaccard_threshold > 0.0 and jaccard_threshold < 1.0:
        list_encoded_data = encoded_data.tolist()
        if not(isinstance(list_encoded_data[0], list)):
            list_encoded_data = [e.tolist() for e in list_encoded_data]
        keep_ids = _remove_overlapping_patterns_jaccard(encoded_patterns, list_encoded_data, 'sequential', jaccard_threshold)
        if len(keep_ids) > 0:
            encoded_patterns = encoded_patterns[keep_ids]
            supports = supports[keep_ids]
        print('DEBUG: # Jaccard thresholded patterns: {}'.format(len(keep_ids)))

    # decode the patterns: array of arrays
    patterns = patterns_decode_from_int(encoded_patterns, decode_dct)
    npa = len(patterns)

    # report patterns
    print('DEBUG: most frequent patterns: {}'.format([(patterns[i], supports[i]) for i in range(min(5, npa))]))
    print('DEBUG: least frequent patterns: {}'.format([(patterns[i], supports[i]) for i in range(max(0, npa-5), npa)]))

    remove_temp_dir(temp_dir)
    if return_encoded:
        return encoded_patterns, encoded_data
    if include_support:
        return patterns, supports
    return patterns

#Len: new 25/04/2019
def mine_minimal_infrequent_itemsets(data, relative_minsup, include_support=False):
    """ Find rare patterns (= minimally frequent patterns)

    parameters
    ----------
    data : numpy array
        Each row is one window of the time series.
    relative_minsup : float
        Relative minimal support for a frequent pattern.
    include_support : bool
        Return the support of the patterns.

    returns
    -------
    patterns : numpy array
        Each element is a frequent pattern.
    supports : numpy array
        Support of each pattern.
    """
    temp_dir = make_temp_dir()

    # reduce precision
    try:
        data = np.around(data, 10)
    except:
        pass

    # encode the data
    encode_dct, encoded_data = data_encode_as_int(data)
    decode_dct = {v: k for k, v in encode_dct.items()}

    # mine the minimal infrequent itemsets
    encoded_patterns, supports = mine_rare_itemsets(encoded_data, relative_minsup, temp_dir)

    # decode the patterns: encoded_patters = array of arrays
    patterns = patterns_decode_from_int(encoded_patterns, decode_dct)
    npa = len(patterns)

    # report patterns
    print('DEBUG: most frequent patterns: {}'.format([(patterns[i], supports[i]) for i in range(min(5, npa))]))
    print('DEBUG: least frequent patterns: {}'.format([(patterns[i], supports[i]) for i in range(max(0, npa-5), npa)]))

    remove_temp_dir(temp_dir)
    if include_support:
        return patterns, supports
    return patterns


################################################################################
# HELPER FUNCTIONS
################################################################################

def _remove_overlapping_patterns_jaccard(encoded_patterns, encoded_data, pattern_type, jaccard_threshold):
    """ Remove the overlapping patterns with Jaccard
        --> expensive cover computation in Cython.
    """
    # compute the cover
    # encoded patterns = list of <class 'numpy.ndarray'> of <class 'numpy.int64'>
    # encoded daa = list of <class 'list'> of <class 'int'>
    # it should be impossible for any pattern to not cover any data!
    # TODO: supports != covers
    covers = cpm.compute_pattern_cover(encoded_patterns, encoded_data, pattern_type, jaccard_threshold)
    N, _ = covers.shape

    # precompute the cover sets
    cover_sets = []
    for i in range(N):
        l = len(np.where(covers[i, :] > 0.0)[0])
        if l == 0:
            print(i, l, supports[i], encoded_patterns[i])
        cover_sets.append(set(np.where(covers[i, :] == 1)[0]))

    # remove overlapping
    keep_ids = []
    for i in range(N):
        keep = True
        for j in range(i+1, N):
            #intersect = np.multiply(covers[i, :], covers[j, :]).sum()
            #intersect = len(set(np.where(covers[i, :] == 1)[0]) & set(np.where(covers[j, :] == 1)[0]))
            intersect = len(cover_sets[i] & cover_sets[j])
            #c1 = covers[i, :].sum()
            #c2 = covers[j, :].sum()
            c1 = len(cover_sets[i])
            c2 = len(cover_sets[j])
            # if both patterns do not cover anything, c1 and c2 are 0! --> avoid zero division error
            if c1 + c2 == 0:
                print('zero cover')
                js = 0.0
            else:
                js = intersect / (c1 + c2 - intersect)
            if js > jaccard_threshold:
                keep = False
                break
        if keep:
            keep_ids.append(i)

    return np.array(keep_ids)
