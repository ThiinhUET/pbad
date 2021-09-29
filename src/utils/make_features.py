"""
Compute weighted occurrences:
- compute a 'soft' match between each row and each pattern, e.g.
        the itemset pattern {'0.1','0.4'} matches exactly with the window  ['0.1','0.1','0.1','0.1','0.1' '0.4'], that is weight is 1.0
                                          if 'almost' matches with window  ['0.1','0.1','0.1','0.1','0.1' '0.3'], that is weight is 1.0 - 0.1 (distance of 0.4 and 0.3)
        for sequential patterns
"""

import numpy as np

# import cython functions
try:
    from cython_utils import cpatternm as cpm
except ImportError:
    from src.utils.cython_utils import cpatternm as cpm
    # print('Loading CPATTERNM in PATTERN_MINING from TOP directory.')


################################################################################
# MAKE PATTERN-BASED FEATURES
################################################################################

def make_pattern_based_features(data, patterns, par_lambda=2, distance_formula=1, pattern_type='itemset', data_type='continuous'):
    """ Make the pattern-based features.
        --> utilizes underlying cython code to speed up

    parameters
    ----------
    data : numpy array
        Contains the datapoints (n_datapoints, n_features).
        Contains the event logs.
    patterns : numpy array
        Contains the patterns.
    par_lambda : float
        Lambda in the distance function between pattern and datapoint.
    distance_formula : int
        The distance formula to be used.
    pattern_type : str
        'itemset' or 'sequential_pattern' type.
    data_type : str
        'continuous' or 'logs'

    returns
    -------
    features : numpy array
        Pattern-based features.
    """

    # TODO: checks on the range of the values in the data and the patterns
    # TODO: better error handling
    if len(data) == 0:
        raise Exception('ERROR: no data to match with the patterns')
    npa = len(patterns)
    if npa == 0:
        raise Exception('ERROR: no patterns to match with the data')
    
    # distinction between continuous and event logs
    if data_type == 'continuous':
        n, _ = data.shape
        # cython back-end to make the features
        if pattern_type == 'itemset':
            # presort the patterns and the data
            # this depends on whether we're dealing with continuous data or event logs
            data = np.sort(data, axis=1)
            for i, p in enumerate(patterns):
                patterns[i] = np.sort(p)
        features = cpm.make_pattern_based_features_cython(data, np.array(patterns), n, npa, par_lambda, distance_formula)

    else:  # logs
        n = len(data)
        # cython back-end to make the features
        if pattern_type == 'itemset':
            # presort the patterns and the data
            # this depends on whether we're dealing with continuous data or event logs
            for i, d in enumerate(data):
                data[i] = np.sort(d)
            for i, p in enumerate(patterns):
                patterns[i] = np.sort(p)
        list_data = [d.tolist() for d in data]

        """ TODO: proper implementation in Cython """
        features = cpm.make_pattern_based_features_cython_event_logs(list_data, patterns, n, npa)

    return features
