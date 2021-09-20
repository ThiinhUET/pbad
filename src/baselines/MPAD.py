"""
pattern-based anomaly detection
-------------------------------
Matrixe Profile Anomaly Detection.

Reference:
    Yeh, C. C. M., Zhu, Y., Ulanova, L., Begum, N., Ding, Y., Dau, H. A., Keogh, E. (2016, December).
    Matrix profile I: all pairs similarity joins for time series: a unifying view that includes motifs, discords and shapelets.
    In Data Mining (ICDM), 2016 IEEE 16th International Conference on (pp. 1317-1322). IEEE.

:authors: Vincent Vercruyssen & Len Feremans
:copyright:
    Copyright 2019 KU Leuven, DTAI Research Group.
    Copyright 2019 UAntwerpen, ADReM Data Lab.
:license:

"""

import sys, os, time, math
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.signal as sps

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class MPAD:
    """ Anomaly Detection with the Matrix Profile.

    Parameters
    ----------
    window_size : int
        Window size.

    Note: the MPAD works only on continuous data.
    Note: works directly on the raw (undiscretized) time series data.
    """

    def __init__(self,
                 window_size=100,           # window size
                 verbose=True):             # verbose

        # checks on the parameters
        if not isinstance(window_size, int):
            print('WARNING: `window_size` parameter should be INT, set to 10')
            window_size = 10
        self.window_size = max(1, window_size)  # window size ranges from 1 to inf

        self.verbose = verbose

    def fit_predict(self, continuous_data={}):
        """ Fit to the time series data.
            Predict the anomalies.

        Parameters
        ----------
        continuous_data : dictionary {number: np.array}
            Dictionary containing the windowed continuous time series data.

        Returns
        -------
        y_score : np.array
            Anomaly scores for each window in the data.

        Note: can only handle continuous data.
        """

        return self.fit(continuous_data)._predict()

    def fit(self, continuous_data={}):
        """ Fit to the time series data.

        Parameters
        ----------
        continuous_data : dictionary {number: np.array}
            Dictionary containing the windowed continuous time series data.
        """

        # checks on the input
        continuous_data = self._check_input(continuous_data)
        ncs = len(continuous_data)

        # continuous data: add the scores
        self.y_score = np.zeros(self.ls, dtype=np.float)
        for nr, series in continuous_data.items():
            n = len(series)
            matrix_profile = self._compute_matrix_profile_stomp(series, self.window_size)
            # transform to an anomaly score (1NN distance)
            # the largest distance = the largest anomaly
            # rescale between 0 and 1, this yields the anomaly score
            new_score = (matrix_profile - min(matrix_profile)) / (max(matrix_profile) - min(matrix_profile))
            new_score = np.append(new_score, np.zeros(n-len(matrix_profile), dtype=float))

            self.y_score += new_score

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
        """

        return self.y_score

    def _compute_matrix_profile_stomp(self, T, m):
        """ Compute the matrix profile and profile index for time series T using correct STOMP.

        Parameters
        ----------
        T : np.array(), shape (n_samples)
            The time series data for which to compute the matrix profile.
        m : int
            Length of the query.

        Returns
        -------
        matrix_profile : np.array(), shape (n_samples)
            The matrix profile (distance) for the time series T.

        Note: Includes a fix for straight line time series segments.
        """

        n = len(T)

        # precompute the mean, standard deviation
        s = pd.Series(T)
        data_m = s.rolling(m).mean().values[m-1:n]
        data_s = s.rolling(m).std().values[m-1:n]

        # where the data is zero
        idxz = np.where(data_s < 1e-8)[0]
        data_s[idxz] = 0.0
        idxn = np.where(data_s > 0.0)[0]

        zero_s = False
        if len(idxz) > 0:
            zero_s = True

        # precompute distance to straight line segment of 0s
        slD = np.zeros(n-m+1, dtype=float)
        if zero_s:
            for i in range(n-m+1):
                Tsegm = T[i:i+m]
                Tm = data_m[i]
                Ts = data_s[i]
                if Ts == 0.0:  # data_s is effectively 0
                    slD[i] = 0.0
                else:
                    Tn = (Tsegm - Tm) / Ts
                    slD[i] = np.sqrt(np.sum(Tn ** 2))

        # compute the first dot product
        q = T[:m]
        QT = sps.convolve(T.copy(), q[::-1], 'valid', 'direct')
        QT_first = QT.copy()

        # compute the distance profile
        D = self._compute_fixed_distance_profile(T[:m], QT, n, m, data_m, data_s, data_m[0], data_s[0], slD.copy(), idxz, idxn, zero_s)

        # initialize matrix profile
        matrix_profile = D

        # in-order evaluation of the rest of the profile
        for i in tqdm(range(1, n-m+1, 1), disable=not(self.verbose)):
            # update the dot product
            QT[1:] = QT[:-1] - (T[:n-m] * T[i-1]) + (T[m:n] * T[i+m-1])
            QT[0] = QT_first[i]

            # compute the distance profile: without function calls!
            if data_s[i] == 0.0:  # query_s is effectively 0
                D = slD.copy()
            elif zero_s:
                D[idxn] = np.sqrt(2 * (m - (QT[idxn] - m * data_m[idxn] * data_m[i]) / (data_s[idxn] * data_s[i])))
                nq = (q - data_m[i]) / data_s[i]
                d = np.sqrt(np.sum(nq ** 2))
                D[idxz] = d
            else:
                D = np.sqrt(2 * (m - (QT - m * data_m * data_m[i]) / (data_s * data_s[i])))

            # update the matrix profile
            exclusion_range = (int(max(0, round(i-m/2))), int(min(round(i+m/2+1), n-m+1)))
            D[exclusion_range[0]:exclusion_range[1]] = np.inf

            ix = np.where(D < matrix_profile)[0]
            matrix_profile[ix] = D[ix]
            # matrix_profile = np.minimum(matrix_profile, D)

        return matrix_profile

    def _compute_fixed_distance_profile(self, q, QT, n, m, data_m, data_s, query_m, query_s, slD, idxz, idxn, zero_s):
        """ Compute the fixed distance profile """
        D = np.zeros(n-m+1, dtype=float)

        if query_s == 0.0:  # query_s is effectively 0
            return slD

        if zero_s:
            D[idxn] = np.sqrt(2 * (m - (QT[idxn] - m * data_m[idxn] * query_m) / (data_s[idxn] * query_s)))
            nq = (q - query_m) / query_s
            d = np.sqrt(np.sum(nq ** 2))
            D[idxz] = d
        else:
            D = np.sqrt(2 * (m - (QT - m * data_m * query_m) / (data_s * query_s)))

        return D

    def _compute_matrix_profile_stamp(self, T, m):
        """ Compute the matrix profile and profile index for time series T using STAMP.

        Parameters
        ----------
        T : np.array(), shape (n_samples)
            The time series data for which to compute the matrix profile.
        m : int
            Length of the query.

        Returns
        -------
        matrix_profile : np.array(), shape (n_samples)
            The matrix profile (distance) for the time series T.
        profile_index : np.array(), shape (n_samples)
            The matrix profile index accompanying the matrix profile.

        Note: Uses the STAMP algorithm to compute the matrix profile.
        Note: Includes a fix for straight line time series segments.
        """

        n = len(T)

        # initialize the empty profile and index
        matrix_profile = np.ones(n-m+1) * np.inf

        # precompute the mean, standard deviation
        s = pd.Series(T)
        data_m = s.rolling(m).mean().values[m-1:n]
        data_s = s.rolling(m).std().values[m-1:n]

        # where the data is zero
        idxz = np.where(data_s < 1e-8)[0]
        data_s[idxz] = 0.0
        idxn = np.where(data_s > 0.0)[0]

        zero_s = False
        if len(idxz) > 0:
            zero_s = True

        # precompute distance to straight line segment of 0s
        # brute force distance computation (because the dot_product is zero!)
        # --> this is a structural issue with the MASS algorithm for fast distance computation
        slD = np.zeros(n-m+1, dtype=float)
        if zero_s:
            for i in range(n-m+1):
                Tsegm = T[i:i+m]
                Tm = data_m[i]
                Ts = data_s[i]
                if Ts == 0.0:  # data_s is effectively 0
                    slD[i] = 0.0
                else:
                    Tn = (Tsegm - Tm) / Ts
                    slD[i] = np.sqrt(np.sum(Tn ** 2))

        # random search order for the outer loop
        indices = np.arange(0, n-m+1, 1)
        np.random.shuffle(indices)

        # compute the matrix profile
        if self.verbose: print('Iterations:', len(indices))
        for i, idx in tqdm(enumerate(indices), disable=not(self.verbose)):
            # query for which to compute the distance profile
            query = T[idx:idx+m]

            # normalized distance profile (using MASS)
            D = self._compute_MASS(query, T, n, m, data_m, data_s, data_m[idx], data_s[idx], slD.copy())

            # update the matrix profile (keeping minimum distances)
            # self-join is True! (we only look at constructing the matrix profile for a single time series)
            exclusion_range = (int(max(0, round(idx-m/2))), int(min(round(idx+m/2+1), n-m+1)))
            D[exclusion_range[0]:exclusion_range[1]] = np.inf

            ix = np.where(D < matrix_profile)[0]
            matrix_profile[ix] = D[ix]

        return matrix_profile

    def _compute_MASS(self, query, T, n, m, data_m, data_s, query_m, query_s, slD):
        """ Compute the distance profile using the MASS algorithm.

        Parameters
        ----------
        query : np.array(), shape (self.m)
            Query segment for which to compute the distance profile.
        T : np.array(), shape (n_samples)
            The time series data for which to compute the matrix profile.
        n : int
            Length of time series T.
        m : int
            Length of the query.
        data_f : np.array, shape (n + m)
            FFT transform of T.
        data_m : np.array, shape (n - m + 1)
            Mean of every segment of length m of T.
        data_s : np.array, shape (n - m + 1)
            STD of every segment of length m of T.
        query_m : float
            Mean of the query segment.
        query_s : float
            Standard deviation of the query segment.

        Returns
        -------
        dist_profile : np.array(), shape (n_samples)
            Distance profile of the query to time series T.
        """

        # CASE 1: query is a straight line segment of 0s
        if query_s < 1e-8:
            return slD

        # CASE 2: query is every other possible subsequence
        # compute the sliding dot product
        reverse_query = query[::-1]
        dot_product = sps.fftconvolve(T, reverse_query, 'valid')

        # compute the distance profile without correcting for standard deviation of the main signal being 0
        # since this is numpy, it will result in np.inf if the data_sig is 0
        dist_profile = np.sqrt(2 * (m - (dot_product - m * query_m * data_m) / (query_s * data_s)))

        # correct for data_s being 0
        zero_idxs = np.where(data_s < 1e-8)[0]
        if len(zero_idxs) > 0:
            n_query = (query - query_m) / query_s
            d = np.linalg.norm(n_query - np.zeros(m, dtype=float))
            dist_profile[zero_idxs] = d

        return dist_profile

    def _compute_brute_force_distance_profile(self, query, T, n, m, data_f, data_m, data_s, query_m, query_s):
        """ Compute the brute force distance profile. """

        dist_profile = np.zeros(n-m+1, dtype=float)

        # normalize query
        if query_m < 1e-8:
            n_query = np.zeros(m, dtype=float)
        else:
            n_query = (query - query_m) / query_s

        # compute the distance profile
        for i in range(n-m+1):
            T_segm = T[i:i+m]
            Tm = data_m[i]
            Ts = data_s[i]
            # normalize time series segment
            if Ts < 1e-8:
                T_norm = np.zeros(m, dtype=float)
            else:
                T_norm = (T_segm - Tm) / Ts
            # compute distance
            dist_profile[i] = np.linalg.norm(T_norm - n_query)

        return dist_profile

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
            if len(v.shape) > 1:
                raise Exception('ERROR: the continuous time series should not be windowed (IN CONTRAST TO THE OTHER METHODS)')
            ts_lengths.append(len(v))
        if len(set(ts_lengths)) != 1:
            raise Exception('ERROR: each time series should have the length')

        self.ls = ts_lengths[0]

        return continuous_data
