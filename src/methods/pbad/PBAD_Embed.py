"""
pattern-based anomaly detection
"""

import sys, os, time, math
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from collections import OrderedDict
from ..utils.pattern_mining import mine_non_redundant_itemsets, mine_non_redundant_sequential_patterns
from ..utils.make_features import make_pattern_based_features

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class PBAD_Embed:
    """ Pattern-based Anomaly Detection.

    Parameters
    ----------
    itemset_filenames_cont: TODO
    sp_filenames_cont: TODO
    itemset_filenames_disc: TODO
    sp_filenames_discr: TODO
    
    distance_lambda : float
        Parameter for scaling the distance when computing the similarity score.
    distance_formula : int
        Parameter to determine which formula to use to compute the distance in the score:
        1. Square root of the sum of the distances between the matched elements.
        2. Sum of the distances to the power (1 / distance_lambda) between the matched elements.
    exact_match : bool
        Require an exact match when computing the similarity score.
    anomaly_classifier : object
        The final anomaly detection classifier.
        The classifier should have a fit() and predict() function.
        If not available, Scikit-learns IsolationForest is used.
    verbose : bool
        Verbose.
    """

    def __init__(self,
                 pattern_type='all',                    # the type of pattern to be mined
                 itemset_filenames_cont=None,           # filename of mined itemsets for continuous series
                 sp_filenames_cont=None,
                 itemset_filenames_discrete=None,       # filename of mined itemsets for discrete series
                 sp_filenames_discrete=None,
                 distance_lambda=2.0,                   # scaling the distance in the similarity score
                 distance_formula=1,                    # type of distance formula: 1 = square root of the sum, 2 = sum of the powers
                 exact_match=False,                     # require an exact match when computing similarity score
                 anomaly_classifier=None,               # object with fit() and predict() --> final classifier
                 verbose=True):                         # verbose

        # checks on the input
        self.itemset_filenames_cont = itemset_filenames_cont
        self.sp_filenames_cont = sp_filenames_cont
        self.itemset_filenames_discrete = itemset_filenames_discrete
        self.sp_filenames_discrete = sp_filenames_discrete
        for filenames in [itemset_filenames_cont, sp_filenames_cont, itemset_filenames_discrete, sp_filenames_discrete]:
            if filenames != None:
                for fname in filenames:
                    if not os.path.isfile(fname):
                        raise Exception('Can not read ' + fname)
        if not isinstance(pattern_type, str):
            print('WARNING: `pattern_type` should be a string, set to `all`')
            pattern_type = 'all'
        if not pattern_type in ['all', 'itemset', 'sequential']:
            print('WARNING: `pattern_type` can only be: all, itemset, sequential')
            pattern_type = 'all'
        self.pattern_type = pattern_type
        
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
        
    def read_itemsets(self, filename):
        #e.g. TIPM input like: 
        #pattern,support
        #pc1=3,105
        #pc1=2,99
        #pc1=4,98
        #pc1=1 pc1=2 pc1=3,87
        with open(filename, 'r') as file:
            tipm_lines = file.readlines()
        tipm_lines = [p.strip() for p in tipm_lines if p.strip() != '']
        tipm_lines = tipm_lines[1:]
        patterns = []
        supports = []
        for line in tipm_lines:
            new_s = np.int(int(line.split(',')[1]))
            tokens = line.split(',')[0].split(' ')
            items = []
            for token in tokens:
                items.append(float(token.split('=')[1]))
            new_p = np.array(items).astype(np.float64)
            supports.append(new_s)
            patterns.append(new_p)
        return patterns, supports

    def read_sp(self, filename):
        return self.read_itemsets(filename)

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

        self.PBAD_features = []

        # continuous data
        if ncs > 0:
            tc = time.time()
            self.cont_pattern_dct = OrderedDict({i: {} for i in range(ncs)})
            idx = 0
            for nr, disc_series in continuous_data_discretized.items():
                # mine the patterns
                '''
                if self.pattern_type == 'all':
                    IS_patterns = mine_non_redundant_itemsets(disc_series, self.relative_minsup, self.jaccard_threshold, self.pattern_pruning)
                    SQ_patterns = mine_non_redundant_sequential_patterns(disc_series, self.relative_minsup, self.jaccard_threshold, self.pattern_pruning)
                elif self.pattern_type == 'itemset':
                    IS_patterns = mine_non_redundant_itemsets(disc_series, self.relative_minsup, self.jaccard_threshold, self.pattern_pruning)
                    SQ_patterns = []
                elif self.pattern_type == 'sequential':
                    SQ_patterns = mine_non_redundant_sequential_patterns(disc_series, self.relative_minsup, self.jaccard_threshold, self.pattern_pruning)
                    IS_patterns = []
                else:
                    raise Exception('ERROR PBAD: unknown `pattern_type`')
                
                # remove sequential patterns that do not have the necessary min length
                if self.sequential_minlength > 1 and len(SQ_patterns) > 0:
                    SQ_patterns = [sqp for sqp in SQ_patterns if len(sqp) > self.sequential_minlength]

                pattern_dct = {'itemset': IS_patterns, 'sequential': SQ_patterns}
                '''
                #read patterns from files
                if self.pattern_type == 'all':
                    IS_patterns, supports = self.read_itemsets(self.itemset_filenames_cont[idx])
                    SQ_patterns, supports = self.read_sp(self.sp_filenames_cont[idx])
                elif self.pattern_type == 'itemset':
                    IS_patterns, supports = self.read_itemsets(self.itemset_filenames_cont[idx])
                    SQ_patterns = []
                elif self.pattern_type == 'sequential':
                    SQ_patterns, supports = self.read_sp(self.sp_filenames_cont[idx])
                    IS_patterns = []
                else:
                    raise Exception('ERROR PBAD: unknown `pattern_type`')
               
                pattern_dct = {'itemset': IS_patterns, 'sequential': SQ_patterns}
                
                # construct the features (on the UNdiscretized data)
                series = continuous_data[nr]
                for pn, patterns in pattern_dct.items():
                    if len(patterns) > 0:
                        F_new = make_pattern_based_features(series, patterns, self.distance_lambda, self.distance_formula, pattern_type=pn)
                        self.PBAD_features.append(F_new)

                self.cont_pattern_dct[nr] = pattern_dct
                idx +=1
            if self.verbose:
                print('PBAD - mining patterns + constructing features for continuous data took:', time.time() - tc, 'seconds')

        # event logs
        if nel > 0:
            tc = time.time()
            self.event_pattern_dct = OrderedDict({i: {} for i in range(nel)})
            idx = 0
            for nr, logs in event_data.items():
                # necessary to remove the windows that do not contain events!
                '''
                cleaned_logs = []
                for l in logs:
                    if len(l) > 0:
                        cleaned_logs.append(l)

                # mine the patterns
                if self.pattern_type == 'all':
                    IS_patterns = mine_non_redundant_itemsets(cleaned_logs, self.relative_minsup, self.jaccard_threshold, self.pattern_pruning)
                    SQ_patterns = mine_non_redundant_sequential_patterns(cleaned_logs, self.relative_minsup, self.jaccard_threshold, self.pattern_pruning)
                elif self.pattern_type == 'itemset':
                    IS_patterns = mine_non_redundant_itemsets(cleaned_logs, self.relative_minsup, self.jaccard_threshold, self.pattern_pruning)
                    SQ_patterns = []
                elif self.pattern_type == 'sequential':
                    SQ_patterns = mine_non_redundant_sequential_patterns(cleaned_logs, self.relative_minsup, self.jaccard_threshold, self.pattern_pruning)
                    IS_patterns = []
                else:
                    raise Exception('ERROR PBAD: unknown `pattern_type`')

                # remove sequential patterns that do not have the necessary min length
                if self.sequential_minlength > 1 and len(SQ_patterns) > 0:
                    SQ_patterns = [sqp for sqp in SQ_patterns if len(sqp) > self.sequential_minlength]

                pattern_dct = {'itemset': IS_patterns, 'sequential': SQ_patterns}
                '''
                #read patterns from files
                if self.pattern_type == 'all':
                    IS_patterns = self.read_itemsets(self.itemset_filenames_discrete[idx])
                    SQ_patterns = self.read_sp(self.sp_filenames_discrete[idx])
                elif self.pattern_type == 'itemset':
                    IS_patterns = self.read_itemsets(self.itemset_filenames_discrete[idx])
                    SQ_patterns = []
                elif self.pattern_type == 'sequential':
                    SQ_patterns = self.read_sp(self.sp_filenames_discrete[idx])
                    IS_patterns = []
                else:
                    raise Exception('ERROR PBAD: unknown `pattern_type`')
               
                pattern_dct = {'itemset': IS_patterns, 'sequential': SQ_patterns}
                
                # construct the features
                for pn, patterns in pattern_dct.items():
                    if len(patterns) > 0:
                        F_new = make_pattern_based_features(logs, patterns, self.distance_lambda, self.distance_formula, pattern_type=pn)
                        F_new[F_new < 1.0] = 0.0
                        self.PBAD_features.append(F_new)

                self.event_pattern_dct[nr] = pattern_dct
                idx+=1    
            if self.verbose:
                print('PBAD - mining patterns + constructing features for event data took:', time.time() - tc, 'seconds')

        # concatenate the features
        if len(self.PBAD_features) == 0:
            raise Exception('ERROR PBAD: no pattern-based features were constructed')
        else:
            self.PBAD_features = np.hstack(self.PBAD_features)

        # exact match required or not
        if self.exact_match:
            self.PBAD_features[self.PBAD_features < 1.0] = 0.0

        # drop zero columns
        ix_nonzero = np.where(np.sum(self.PBAD_features, axis=0) > 0.0)[0]
        self.PBAD_features = self.PBAD_features[:, ix_nonzero]
        _, nf = self.PBAD_features.shape

        # train the classifier
        tc = time.time()
        if self.anomaly_classifier is None:
            self.clf = IsolationForest(n_estimators=500, max_samples='auto', max_features=min(50, nf))
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
        if len(continuous_data) == 0 and len(event_data) == 0 or len(continuous_data_discretized) == 0:
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
