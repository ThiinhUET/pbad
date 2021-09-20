"""
pattern-based anomaly detection
-------------------------------
Pattern Anomaly Value.

Reference:
    Chen, X. Y., & Zhan, Y. Y. (2008).
    Multi-scale anomaly detection algorithm based on infrequent pattern of time series.
    Journal of Computational and Applied Mathematics, 214(1), 227-237.

:authors: Vincent Vercruyssen & Len Feremans
:copyright:
    Copyright 2019 KU Leuven, DTAI Research Group.
    Copyright 2019 UAntwerpen, ADReM Data Lab.
:license:

"""

import sys, os, time, math
import numpy as np

from collections import Counter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class PAV:
    """ Pattern Anomaly Value

    Parameters
    ----------

    """

    def __init__(self,
                 verbose=True):             # verbose

        self.verbose = verbose

    def fit_predict(self, continuous_data={}, window_size=1, window_incr=1):
        """ Fit to the time series data.
            Predict the anomalies.

        Parameters
        ----------
        continuous_data : dictionary {number: np.array}
            Dictionary containing the windowed continuous time series data.
        window_size : int
            Size of the windows in the windowed time series.
        window_incr : int
            Increment of the windows in the windowed time series.

        Returns
        -------
        y_score : np.array
            Anomaly scores for each window in the data.

        Note: can only handle continuous data.
        """

        return self.fit(continuous_data, window_size, window_incr)._predict()

    def fit(self, continuous_data={}, window_size=1, window_incr=1):
        """ Fit to the time series data.

        Parameters
        ----------
        continuous_data : dictionary {number: np.array}
            Dictionary containing the windowed continuous time series data.
        window_size : int
            Size of the windows in the windowed time series.
        window_incr : int
            Increment of the windows in the windowed time series.
        """

        self.window_size = int(window_size)
        self.window_incr = int(window_incr)

        # checks on the input
        continuous_data = self._check_input(continuous_data)
        ncs = len(continuous_data)

        self.y_score = np.zeros(self.nw, dtype=np.float)

        # compute the patterns
        for nr, series in continuous_data.items():
            # reconstruct the original series
            if self.window_incr == self.window_size:
                ts = series.flatten(order='C')
                ns = len(ts)
            elif self.window_incr < self.window_size:
                ts = np.concatenate((series[:-1,:self.window_incr].flatten(order='C'), series[-1,:]))
                ns = len(ts)
                assert ns == int((len(series) - 1) * self.window_incr + self.window_size), 'ERROR PAV: could not faithfully reconstruct original time series'
            else:
                raise Exception('ERROR PAV: cannot fully reconstruct the unwindowed time series --> would skew pattern support')

            # create the patterns: precision = 3 numbers after comma
            fslope = lambda x1, y1: '{0:.3f}'.format(y1 - x1)
            flength = lambda x1, y1: '{0:.3f}'.format(np.sqrt((x1 - y1) ** 2), 3)
            str_patterns = np.array([fslope(ts[i], ts[i+1]) + flength(ts[i], ts[i+1]) for i in range(ns - 1)])  # entire series!

            # normalized support for the patterns
            pattern_dict = Counter(str_patterns)
            maxi = max(pattern_dict.values())
            mini = min(pattern_dict.values())
            for k, v in pattern_dict.items():
                pattern_dict[k] = (v - mini) / (maxi - mini)

            # anomaly score
            ascore = np.array([1.0 - pattern_dict[sp] for sp in str_patterns])

            # anomaly score for each window: simply sum the scores of individual points
            pav_scores = np.zeros(len(series), dtype=np.float)
            for i in range(len(series)):
                sum_av = np.sum(ascore[i*self.window_incr:i*self.window_incr+self.window_size])
                pav_scores[i] = sum_av

            # sum the scores for the different series
            norm_scores = (pav_scores - min(pav_scores)) / (max(pav_scores) - min(pav_scores))
            self.y_score = self.y_score + norm_scores

        self.y_score = self.y_score / ncs

        return self

    def predict(self, continuous_data={}):
        """ Predict the anomalies.

        Parameters
        ----------
        continuous_data : dictionary {number: np.array}
            Dictionary containing the windowed continuous time series data.

        Returns
        -------
        y_score : np.array
            Anomaly scores for each window in the data.

        Note: expects the same window size and increment as fit().
        """

        # checks on the input
        sys.exit('ERROR: predict() to be implemented --> use _predict()')

    def _predict(self):
        """ Predict the anomalies.

        Returns
        -------
        y_score : np.array
            Anomaly scores for each window in the data.

        Note: uses the data passed to the fit() method.
        Note: expects the same window size and increment as fit().
        """

        return self.y_score

    def _check_input(self, continuous_data):
        """ Check if the input has the right format. """

        # input should be given
        if len(continuous_data) == 0:
            raise Exception('ERROR: no input given')

        # remaining checks
        if not isinstance(continuous_data, dict):
            sys.exit('ERROR: `continuous_data` should be a dictionary containing the number + series')
        ts_lengths = []
        for k, v in continuous_data.items():
            if not isinstance(k, int):
                raise Exception('ERROR: the continuous time series should be numbered (INT)')
            if not isinstance(v, np.ndarray):
                raise Exception('ERROR: continuous time series data should be numpy arrays')
            ts_lengths.append(len(v))
        if len(set(ts_lengths)) != 1:
            raise Exception('ERROR: each time series should have the same number of windows')

        self.nw = ts_lengths[0]

        return continuous_data
