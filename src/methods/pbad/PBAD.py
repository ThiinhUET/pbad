"""
pattern-based anomaly detection
-------------------------------
Pattern-based anomaly detector.
"""

import sys, os, time, math
import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import IsolationForest
from collections import OrderedDict
from utils.pattern_mining import mine_non_redundant_itemsets, mine_non_redundant_sequential_patterns
from utils.make_features import make_pattern_based_features

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class PBAD:
    """ Pattern-based Anomaly Detection.

    Parameters
    ----------
    pattern_type : str
        Type of pattern to be used: 'all', 'itemset', 'sequential'.
    relative_minsup : float
        Relative minimum support for the frequent patterns.
    jaccard_threshold : float
        Jaccard threshold to prune the mined patterns.
    pattern_pruning : str
        Pattern pruning strategy: 'closed', 'maximal'.
    sequential_minlength : int
        Minimum required length for the sequential patterns.
    distance_lambda : float
        Parameter for scaling the distance when computing the similarity score.
    distance_formula : int
        Parameter to determine which formula to use to compute the distance in the score:
        1. Square root of the sum of the distances between the matched elements.
        2. Sum of the distances to the power (1 / distance_lambda) between the matched elements.
    exact_match : bool
        Require an exact match when computing the similarity score.
    pattern_match_discrete : bool
        Match the patterns with the discritized time series or the continuous time series.
    anomaly_classifier : object
        The final anomaly detection classifier.
        The classifier should have a fit() and predict() function.
        If not available, Scikit-learns IsolationForest is used.
    verbose : bool
        Verbose.
    """

    def __init__(self,
                 pattern_type='all',            # the type of pattern to be mined {all, itemset, sequential, raw}
                 relative_minsup=0.01,          # relative minimum support to determine frequent patterns
                 jaccard_threshold=0.9,         # jaccard threshold to prune overlapping patterns
                 pattern_pruning='closed',      # pattern pruning strategy when mining patterns
                 sequential_minlength=1.0,      # minimum required length for sequential patterns
                 distance_lambda=2.0,           # scaling the distance in the similarity score
                 distance_formula=2,            # type of distance formula: 1 = square root of the sum, 2 = sum of the powers
                 exact_match=False,             # require an exact match when computing similarity score
                 pattern_match_discrete=False,  # patterns match with discritized time series (True) or continuous time series (False)
                 anomaly_classifier=None,       # object with fit() and predict() --> final classifier
                 verbose=True):                 # verbose

        # checks on the input
        if not isinstance(pattern_type, str):
            print('WARNING: `pattern_type` should be a string, set to `all`')
            pattern_type = 'all'
        if not pattern_type in ['all', 'itemset', 'sequential', 'raw']:
            print('WARNING: `pattern_type` can only be: all, itemset, sequential, raw')
            pattern_type = 'all'
        self.pattern_type = pattern_type

        if not isinstance(relative_minsup, float):
            print('WARNING: `relative_minsup` should be a float, set to 0.01')
            relative_minsup = 0.01
        self.relative_minsup = min(1.0, max(0.0, relative_minsup))  # between 0 and 1

        if not isinstance(jaccard_threshold, float):
            print('WARNING: `jaccard_threshold` should be a float, set to 0.9')
            jaccard_threshold = 0.9
        self.jaccard_threshold = min(1.0, max(0.0, jaccard_threshold))  # between 0 and 1

        if not isinstance(pattern_pruning, str):
            print('WARNING: `pattern_pruning` should be a string, set to `closed`')
            pattern_pruning = 'closed'
        if not pattern_pruning in ['closed', 'maximal']:
            print('WARNING: `pattern_pruning` can only be: closed, maximal')
            pattern_pruning = 'closed'
        self.pattern_pruning = pattern_pruning

        if not isinstance(sequential_minlength, float):
            print('WARNING: `sequential_minlength` should be an int, set to 2')
            sequential_minlength = 2
        self.sequential_minlength = max(1, sequential_minlength)  # between 1 and inf

        if not isinstance(distance_lambda, float):
            print('WARNING: `distance_lambda` should be an float, set to 2.0')
            distance_lambda = 2.0
        self.distance_lambda = max(0.0, distance_lambda)  # between 0 and inf

        if not(isinstance(distance_formula, int) or distance_formula in [1, 2]):
            print('WARNING: `distance_formula` should be an int (either 1 or 2), set to 1')
            distance_formula = 1
        self.distance_formula = distance_formula

        if not isinstance(exact_match, bool):
            print('WARNING: `exact_match` should be True or False, set to False')
            exact_match = False
        self.exact_match = exact_match

        if not isinstance(pattern_match_discrete, bool):
            print('WARNING: `pattern_match_discrete` should be True or False, set to False')
            pattern_match_discrete = False
        self.pattern_match_discrete = pattern_match_discrete

        if not isinstance(anomaly_classifier, object):
            print('WARNING: `anomaly_classifier` should be an object with fit() and predict(), using scikit-learn IsolationForest')
            self.anomaly_classifier = None
            # set the parameters later
        else:
            fit_fnc = getattr(anomaly_classifier, 'fit', None)
            has_fit = callable(fit_fnc)
            pre_fnc = getattr(anomaly_classifier, 'predict', None)
            has_pre = callable(pre_fnc)
            if not has_fit:
                print('WARNING: `anomaly_classifier` has no fit() function, using scikit-learn IsolationForest')
                self.anomaly_classifier = None
            elif not has_pre:
                print('WARNING: `anomaly_classifier` has no predict() function, using scikit-learn IsolationForest')
                self.anomaly_classifier = None
            else:
                self.anomaly_classifier = anomaly_classifier

        self.verbose = verbose

    def fit_predict(self, continuous_data={}, continuous_data_discretized={}, event_data={}):
        """ Fit PBAD to the time series data.
            Predict the anomalies.

        Parameters
        ----------
        continuous_data : dictionary {number: np.array}
            Dictionary containing the windowed continuous time series data.
        continuous_data_discretized : dictionary {number: np.array}
            Dictionary containing the discretized windowed continuous time series data.
        event_data : dictionary {number: np.array}
            Dictionary containing the windowed event data.

        Returns
        -------
        y_score : np.array
            Anomaly scores for each window in the data.
        """

        return self.fit(continuous_data, continuous_data_discretized, event_data)._predict()

    def fit(self, continuous_data={}, continuous_data_discretized={}, event_data={}):
        """ Fit PBAD to the time series data.

        Parameters
        ----------
        continuous_data : dictionary {number: np.array}
            Dictionary containing the windowed continuous time series data.
        continuous_data_discretized : dictionary {number: np.array}
            Dictionary containing the discretized windowed continuous time series data.
        event_data : dictionary {number: np.array}
            Dictionary containing the windowed event data.
        """

        # checks on the input
        continuous_data, continuous_data_discretized, event_data = self._check_input(continuous_data, continuous_data_discretized, event_data)
        ncs = len(continuous_data_discretized)
        nel = len(event_data)

        # match with continuous or discrete
        if self.pattern_match_discrete:
            continuous_data = continuous_data_discretized.copy()

        self.PBAD_features = []

        # continuous data
        if ncs > 0:
            tc = time.time()
            self.cont_pattern_dct = OrderedDict({i: {} for i in range(ncs)})
            for nr, disc_series in continuous_data_discretized.items():
                # mine the patterns
                if self.pattern_type == 'all':
                    IS_patterns = mine_non_redundant_itemsets(disc_series, self.relative_minsup, self.jaccard_threshold, self.pattern_pruning, return_encoded=False)
                    SQ_patterns = mine_non_redundant_sequential_patterns(disc_series, self.relative_minsup, self.jaccard_threshold, self.pattern_pruning, return_encoded=False)
                elif self.pattern_type == 'itemset':
                    IS_patterns = mine_non_redundant_itemsets(disc_series, self.relative_minsup, self.jaccard_threshold, self.pattern_pruning, return_encoded=False)
                    SQ_patterns = []
                elif self.pattern_type == 'sequential':
                    SQ_patterns = mine_non_redundant_sequential_patterns(disc_series, self.relative_minsup, self.jaccard_threshold, self.pattern_pruning, return_encoded=False)
                    IS_patterns = []
                elif self.pattern_type == 'raw':
                    SQ_patterns = []
                    IS_patterns = []
                else:
                    raise Exception('ERROR PBAD: unknown `pattern_type`')

                # remove sequential patterns that do not have the necessary min length
                # if only sequential patterns of length 1 are available, skip this action
                if self.sequential_minlength > 1 and len(SQ_patterns) > 0:
                    new_SQ_patterns = [sqp for sqp in SQ_patterns if len(sqp) > self.sequential_minlength]
                    # TODO: improve this behavior
                    if not(len(new_SQ_patterns) == 0):
                        SQ_patterns = new_SQ_patterns

                pattern_dct = {'itemset': IS_patterns, 'sequential': SQ_patterns}

                # construct the features (on the UNdiscretized data)
                series = continuous_data[nr]
                if not(self.pattern_type == 'raw'):
                    for pn, patterns in pattern_dct.items():
                        if len(patterns) > 0:
                            F_new = make_pattern_based_features(series, patterns, self.distance_lambda,
                                self.distance_formula, pattern_type=pn, data_type='continuous')
                            self.PBAD_features.append(F_new)
                else:
                    self.PBAD_features.append(series)

                self.cont_pattern_dct[nr] = pattern_dct

            if self.verbose:
                print('PBAD - mining patterns + constructing features for continuous data took:', time.time() - tc, 'seconds')

        # event logs
        if nel > 0:
            tc = time.time()
            self.event_pattern_dct = OrderedDict({i: {} for i in range(nel)})
            for nr, logs in event_data.items():
                """ TODO: remove logs that are empty. It is a choice:
                    1. include an empty log as an item
                    2. exclude empty logs (i.e, no patterns)
                    
                    This is a detail.
                """

                # mine the patterns: operates on the encoded logs + returns the encoded logs!
                if self.pattern_type == 'all':
                    IS_patterns, IS_encoded_logs = mine_non_redundant_itemsets(logs, self.relative_minsup, self.jaccard_threshold, self.pattern_pruning, return_encoded=True)
                    SQ_patterns, SQ_encoded_logs = mine_non_redundant_sequential_patterns(logs, self.relative_minsup, self.jaccard_threshold, self.pattern_pruning, return_encoded=True)
                elif self.pattern_type == 'itemset':
                    IS_patterns, IS_encoded_logs = mine_non_redundant_itemsets(logs, self.relative_minsup, self.jaccard_threshold, self.pattern_pruning, return_encoded=True)
                    SQ_patterns = []
                elif self.pattern_type == 'sequential':
                    SQ_patterns, SQ_encoded_logs = mine_non_redundant_sequential_patterns(logs, self.relative_minsup, self.jaccard_threshold, self.pattern_pruning, return_encoded=True)
                    IS_patterns = []
                elif self.pattern_type == 'raw':
                    raise Exception('ALGORITHMIC ERROR: event logs do not work yet with `pattern_type` raw')
                else:
                    raise Exception('ERROR PBAD: unknown `pattern_type`')

                # remove sequential patterns that do not have the necessary min length
                if self.sequential_minlength > 1 and len(SQ_patterns) > 0:
                    SQ_patterns = [sqp for sqp in SQ_patterns if len(sqp) > self.sequential_minlength]

                pattern_dct = {'itemset': IS_patterns, 'sequential': SQ_patterns}

                # construct the features: also operates on the encoded logs
                # --> checks exact match on the events
                for pn, patterns in pattern_dct.items():
                    if len(patterns) > 0:
                        if pn == 'itemset':
                            # make features with: patterns (encoded) and IS_encoded_logs
                            F_new = make_pattern_based_features(IS_encoded_logs, patterns, self.distance_lambda, self.distance_formula, pattern_type=pn, data_type='logs')
                        else:
                            # make features with: patterns (encoded) and SQ_encoded_logs
                            F_new = make_pattern_based_features(SQ_encoded_logs, patterns, self.distance_lambda, self.distance_formula, pattern_type=pn, data_type='logs')
                        self.PBAD_features.append(F_new)

                self.event_pattern_dct[nr] = pattern_dct

            if self.verbose:
                print('PBAD - mining patterns + constructing features for event data took:', time.time() - tc, 'seconds')

        for i in range(len(self.PBAD_features)):
            print(self.PBAD_features[i].shape)

        # concatenate the features
        if len(self.PBAD_features) == 0:
            raise Exception('ERROR PBAD: no pattern-based features were constructed')
        elif len(self.PBAD_features) == 1:
            self.PBAD_features = self.PBAD_features[0]
        else:
            self.PBAD_features = np.hstack(self.PBAD_features)

        # exact match required or not
        if self.exact_match:
            self.PBAD_features[self.PBAD_features < 1.0] = 0.0

        """ It could be at this point that there are not enough patterns found
            (because of a combination of the data and the settings to preprocess).
            Then, as a result, not enough patterns are found (sometimes even only 1)
            and no meaningful features are generated from the time series.

            Currently, PBAD will fail because the IsolationForest algorithm does not
            work with only one feature when there is no distinct values.

            The solution is to rerun the algorithm with different settings
            (e.g., higher relative minimum support and `closed` instead of `maximal` patterns).
            This could yield meaningful patterns that can then be used to construct features.        
        """

        # drop zero columns
        ix_nonzero = np.where(np.sum(self.PBAD_features, axis=0) > 0.0)[0]
        self.PBAD_features = self.PBAD_features[:, ix_nonzero]
        _, nf = self.PBAD_features.shape
        
        # train the classifier
        tc = time.time()
        if self.anomaly_classifier is None:
            """ 
            UNIVARIATE:
                Max samples is set higher (1500) because there are many features.
                Otherwise, random data points could be split off by a lucky split
                in the high-dimensional data set.
                For the same reason, limit the number of features to grow each tree.

            MULTIVARIATE:
                Smaller time series, less patterns (because higher alphabet size),
                auto-find the number of samples required to build the forest.
            """
            self.clf = IsolationForest(n_estimators=500, max_samples=1500, max_features=min(50, nf))
        else:
            self.clf = self.anomaly_classifier
        self.clf.fit(self.PBAD_features)
        if self.verbose:
            print('PBAD - training classifier took:', time.time() - tc, 'seconds')

        return self

    def _predict(self):
        """ Predict the anomalies.

        Returns
        -------
        y_score : np.array
            Anomaly scores for each window in the data.

        Note: uses the data passed to the fit() method.
        """

        # TODO: fix when function contains predict() method
        y_score = self.clf.decision_function(self.PBAD_features) * -1
        y_score = (y_score - min(y_score)) / (max(y_score) - min(y_score))

        return y_score

    def _check_input(self, continuous_data, continuous_data_discretized, event_data):
        """ Check if the input has the right format. """

        # input should be given
        if len(continuous_data) == 0 and len(event_data) == 0 and len(continuous_data_discretized) == 0:
            raise Exception('ERROR PBAD: no input given')

        # remaining checks
        if not isinstance(continuous_data, dict):
            raise Exception('ERROR PBAD: `continuous_data` should be a dictionary containing the number + series')
        if not isinstance(continuous_data_discretized, dict):
            raise Exception('ERROR PBAD: `continuous_data_discretized` should be a dictionary containing the number + series')
        if not isinstance(event_data, dict):
            raise Exception('ERROR PBAD: `event_data` should be a dictionary containing the number + event log')
        if not(len(continuous_data) == len(continuous_data_discretized)):
            raise Exception('ERROR PBAD: `continuous_data` and `continuous_data_discretized` should contain the same amount of series')
        ts_lengths = []
        for k, v in continuous_data.items():
            if not isinstance(k, int):
                raise Exception('ERROR PBAD: the continuous time series should be numbered (INT)')
            if not isinstance(v, np.ndarray):
                raise Exception('ERROR PBAD: continuous time series data should be numpy arrays')
            ts_lengths.append(len(v))
        for k, v in continuous_data_discretized.items():
            if not isinstance(k, int):
                raise Exception('ERROR PBAD: the continuous discretized time series should be numbered (INT)')
            if not isinstance(v, np.ndarray):
                raise Exception('ERROR PBAD: continuous discretized time series data should be numpy arrays')
            ts_lengths.append(len(v))
        for k, v in event_data.items():
            if not isinstance(k, int):
                raise Exception('ERROR PBAD: the event logs should be numbered (INT)')
            if not(isinstance(v, np.ndarray) or isinstance(v, list)):
                raise Exception('ERROR PBAD: event log data should be list of arrays')
            # event logs are assumed to be encoded as strings
            ts_lengths.append(len(v))
        if len(set(ts_lengths)) != 1:
            print(set(ts_lengths))
            raise Exception('ERROR PBAD: each time series should have the same number of windows')

        self.nw = ts_lengths[0]

        return continuous_data, continuous_data_discretized, event_data
