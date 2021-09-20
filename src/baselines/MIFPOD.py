"""
pattern-based anomaly detection
-------------------------------
Minimally infrequent pattern outlier detection.

Reference:
    Hemalatha, C. S., Vaidehi, V., & Lakshmi, R. (2015).
    Minimal infrequent pattern based approach for mining outliers in data streams.
    Expert Systems with Applications, 42(4), 1998-2012.

:authors: Vincent Vercruyssen & Len Feremans
:copyright:
    Copyright 2019 KU Leuven, DTAI Research Group.
    Copyright 2019 UAntwerpen, ADReM Data Lab.
:license:

"""

import sys, os, time, math
import pandas as pd
import numpy as np

from collections import Counter, OrderedDict
from utils.pattern_mining import mine_minimal_infrequent_itemsets

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class MIFPOD:
    """ Minimally infrequent pattern outlier detection.

    Parameters
    ----------
    relative_minsup : float
        Relative minimum support for the frequent patterns.
    verbose : bool
        Verbose.
    """

    def __init__(self,
                 relative_minsup=0.01,      # relative minimum support to determine frequent patterns
                 verbose=True):             # verbose

        # checks on the input
        if not isinstance(relative_minsup, float):
            print('WARNING: `relative_minsup` should be a float, set to 0.01')
            relative_minsup = 0.01
        self.relative_minsup = min(1.0, max(0.0, relative_minsup))  # between 0 and 1

        self.verbose = verbose

    def fit_predict(self, continuous_data={}, event_data={}):
        """ Fit MIFPOD to the time series data.
            Predict the anomalies.

        Parameters
        ----------
        continuous_data : dictionary {number: np.array}
            Dictionary containing the windowed continuous time series data.
        event_data : dictionary {number: np.array}
            Dictionary containing the windowed event data.

        Returns
        -------
        y_score : np.array
            Anomaly scores for each window in the data.
        """

        return self.fit(continuous_data, event_data).predict(continuous_data, event_data)

    def fit(self, continuous_data={}, event_data={}):
        """ Fit to the time series data.

        Parameters
        ----------
        continuous_data : dictionary {number: np.array}
            Dictionary containing the windowed continuous time series data.
        event_data : dictionary {number: np.array}
            Dictionary containing the windowed event data.
        """

        # checks on the input
        continuous_data, event_data = self._check_input(continuous_data, event_data)
        ncs = len(continuous_data)
        nel = len(event_data)

        # continuous data
        self.cont_pattern_dct = OrderedDict({i: {} for i in range(ncs)})
        if ncs > 0:
            for nr, series in continuous_data.items():
                # mine the patterns
                IS_patterns, IS_supports = mine_minimal_infrequent_itemsets(series, self.relative_minsup, include_support=True)
                self.cont_pattern_dct[nr] = {'patterns': IS_patterns, 'support': IS_supports}

        # event logs
        self.event_pattern_dct = OrderedDict({i: {} for i in range(nel)})
        if nel > 0:
            for nr, logs in event_data.items():
                # mine the patterns
                IS_patterns, IS_supports = mine_minimal_infrequent_itemsets(series, self.relative_minsup, include_support=True)
                self.event_pattern_dct[nr] = {'patterns': IS_patterns, 'support': IS_supports}

        return self

    def predict(self, continuous_data={}, event_data={}):
        """ Predict the anomalies.

        Parameters
        ----------
        continuous_data : dictionary {number: np.array}
            Dictionary containing the windowed continuous time series data.
        event_data : dictionary {number: np.array}
            Dictionary containing the windowed event data.

        Returns
        -------
        y_score : np.array
            Anomaly scores for each window in the data.
        """

        # checks on the input
        continuous_data, event_data = self._check_input(continuous_data, event_data)
        ncs = len(continuous_data)
        nel = len(event_data)

        y_score = np.zeros(self.nw, dtype=np.float)

        # continuous data
        if len(self.cont_pattern_dct) > 0:
            for nr, series in continuous_data.items():
                IS_patterns = self.cont_pattern_dct[nr]['patterns']
                IS_supports = self.cont_pattern_dct[nr]['support']
                # compute the FPOF score
                new_score = self._compute_mifpod_score(series, IS_patterns, IS_supports)
                if np.sum(new_score) < 1e-8 or abs(np.sum(new_score) - len(new_score)) < 1e-8:
                    print('WARNING: the mifpod scores are all ~0.0 or ~1.0 --> run with different preprocessing settings')
                new_score = (new_score - min(new_score)) / (max(new_score) - min(new_score))
                y_score += new_score

        # event logs
        if len(self.event_pattern_dct) > 0:
            for nr, logs in event_data.items():
                IS_patterns = self.event_pattern_dct[nr]['patterns']
                IS_supports = self.event_pattern_dct[nr]['support']
                # compute the FPOF score
                new_score = self._compute_mifpod_score(logs, IS_patterns, IS_supports)
                if np.sum(new_score) < 1e-8 or abs(np.sum(new_score) - len(new_score)) < 1e-8:
                    print('WARNING: the mifpod scores are all ~0.0 or ~1.0 --> run with different preprocessing settings')
                new_score = (new_score - min(new_score)) / (max(new_score) - min(new_score))
                y_score += new_score

        y_score = y_score / (ncs + nel)

        return y_score

    def _compute_mifpod_score(self, data, patterns, supports):
        """ Compute the FPOF score for each data point using the patterns. """

        n = len(data)
        npa = len(patterns)

        # TODO: make this faster? (Cython...)
        mifpof_score = np.zeros(n, dtype=np.float)
        for i, w in enumerate(data):
            # 1. compute TWF and MIPDF
            twf = 0
            mipdf = 0.0
            for j, mip in enumerate(patterns):
                contained = self._contains_pattern(w, mip)
                if contained:
                    twf += len(mip)
                    mipdf += supports[j]
            twf = twf / npa

            # 2. compute the MIFPOF score
            mifpof_score[i] = 1.0 - twf * mipdf

        return mifpof_score

    def _contains_pattern(self, x, pattern):
        """ Check if transaction x contians the pattern: hard match! """
        return all(e in x for e in pattern)

    def _check_input(self, continuous_data, event_data):
        """ Check if the input has the right format. """

        # input should be given
        if len(continuous_data) == 0 and len(event_data) == 0:
            raise Exception('ERROR: no input given')

        # remaining checks
        if not isinstance(continuous_data, dict):
            raise Exception('ERROR: `continuous_data` should be a dictionary containing the number + series')
        if not isinstance(event_data, dict):
            raise Exception('ERROR: `event_data` should be a dictionary containing the number + event log')
        ts_lengths = []
        for k, v in continuous_data.items():
            if not isinstance(k, int):
                raise Exception('ERROR: the continuous time series should be numbered (INT)')
            if not isinstance(v, np.ndarray):
                raise Exception('ERROR: continuous time series data should be numpy arrays')
            ts_lengths.append(len(v))
        for k, v in event_data.items():
            if not isinstance(k, int):
                raise Exception('ERROR: the event logs should be numbered (INT)')
            if not(isinstance(v, np.ndarray) or isinstance(v, list)):
                raise Exception('ERROR: event log data should be list of arrays')
            # event logs are assumed to be encoded as strings
            ts_lengths.append(len(v))
        if len(set(ts_lengths)) != 1:
            raise Exception('ERROR: each time series should have the same number of windows')

        self.nw = ts_lengths[0]

        return continuous_data, event_data
