"""
pattern-based anomaly detection
-------------------------------
PreProcessor for:
    1. continuous time series
    2. event logs

Preprocess the data:
    - remove extreme values
    - scaling of the data
    - smooth the data
    - sample the data
    - discretize the data
    - bin and window the data

"""

import sys, os, math
import numpy as np

from collections import Counter


class PreProcessor:
    """ Preprocessor for:
        1. continuous time series
        2. discrete event logs
    """

    def __init__(self,
                 remove_extremes=True,  # remove extreme values
                 minmax_scaling=False,  # min-max scaling of the data
                 add_scaling=True,      # rescale the data, data are always scaled between 0 and 1
                 scaler=1.0,            # value to rescale the data with
                 capvalue=1.0,          # maximum value to cap the data (values > capvalue are set to capvalue)
                 window_size=10,        # number of bins of the time series in one window
                 window_incr=10,        # number of bins between every window
                 bin_size=1,            # number of measurements in one bin
                 subsample=1,           # subsample the data (1 = no subsampling, x > 1 = skip every x values)
                 smoothing=False,       # smooth the time series first, moving average smoothing
                 smooth_incr=3,         # smoothing window with moving average smoothing (should be uneven)
                 discretize=True,       # discretize the data
                 alphabet_size=30,      # size of the alphabet to discretize
                 label_scheme=1,        # labeling scheme for the data (purely for running experiments)
                 verbose=True):         # verbose

        # checks on the input values
        if not isinstance(remove_extremes, bool):
            print('WARNING: `remove_extremes` parameter should be BOOL, set to FALSE')
            remove_extremes = True
        self.remove_extremes = remove_extremes

        if not isinstance(minmax_scaling, bool):
            print('WARNING: `minmax_scaling` parameter should be BOOL, set to FALSE')
            minmax_scaling = True
        self.minmax_scaling = minmax_scaling

        if not isinstance(add_scaling, bool):
            print('WARNING: `add_scaling` parameter should be BOOL, set to TRUE')
            add_scaling = True
        self.add_scaling = add_scaling

        if not isinstance(scaler, float):
            print('WARNING: `scaler` parameter should be FLOAT, set to 1.0')
            scaler = 1.0
        self.scaler = max(0.1, scaler)  # scaler ranges from 0.1 to inf

        if not isinstance(capvalue, float):
            print('WARNING: `capvalue` parameter should be FLOAT, set to 1.0')
            capvalue = 1.0
        self.capvalue = max(0.1, capvalue)  # capvalue ranges from 0.1 to inf

        if not isinstance(window_size, int):
            print('WARNING: `window_size` parameter should be INT, set to 10')
            window_size = 10
        self.window_size = max(1, window_size)  # window size ranges from 1 to inf

        if not isinstance(window_incr, int):
            print('WARNING: `window_incr` parameter should be INT, set to 10')
            window_incr = 10
        self.window_incr = max(1, window_incr)  # window increment ranges from 1 to inf

        if not isinstance(bin_size, int):
            print('WARNING: `bin_size` parameter should be INT, set to 1')
            bin_size = 1
        self.bin_size = max(1, bin_size)  # bin size ranges from 1 to inf

        if not isinstance(subsample, int):
            print('WARNING: `subsample` parameter should be INT, set to 1')
            subsample = 1
        self.subsample = max(1, subsample)  # bin size ranges from 1 to inf

        if not isinstance(smoothing, bool):
            print('WARNING: `smoothing` parameter should be BOOL, set to TRUE')
            smoothing = True
        self.smoothing = smoothing

        if not isinstance(smooth_incr, int):
            print('WARNING: `smooth_incr` parameter should be INT, set to 3')
            smooth_incr = 3
        elif smooth_incr % 2 == 0:
            print('WARNING: `smooth_incr` parameter should be UNEVEN, adding 1')
            smooth_incr = smooth_incr + 1
        if smooth_incr < 3:
            print('WARNING: `smooth_incr` should be at least 3, set to 3')
        self.smooth_incr = max(3, smooth_incr)  # smoothing increment in [3, 5, ...]

        if not isinstance(discretize, bool):
            print('WARNING: `smoothing` parameter should be BOOL, set to TRUE')
            discretize = True
        self.discretize = discretize

        if not isinstance(alphabet_size, int):
            print('WARNING: `alphabet_size` parameter should be INT, set to 30')
            alphabet_size = 30
        self.alphabet_size = max(1, alphabet_size)  # alphabet size ranges from 1 to inf

        if not isinstance(label_scheme, int):
            print('WARNING: `label_scheme` parameter should be INT, set to 1')
            label_scheme = 1
        self.label_scheme = label_scheme

        self.verbose = bool(verbose)

    def preprocess(self, continuous_series={}, event_logs={}, labels=np.array([]), output_location='', event_log_placeholder='', return_undiscretized=False):
        """ Preprocess the data + check the validity of the data.
            If labels are given, they are immediately converted to match the data.
            Write the data to a file if specified.

        parameters
        ----------
        continuous_series : dictionary {number: np.array}
            Dictionary containing the continuous time series.
        event_logs : dictionary {number: np.array}
            Dictionary containing the event logs.
        labels : np.array
            Label information if available.
        output_location : str
            If specified, the output will be written to this location.
        event_log_placeholder : str
            The placeholder in the event log when no event occurs.
            Multiple events can occur simultaneously.
        return_undiscretized : bool
            Return the undiscretized data (after all other preprocessing steps).

        returns
        -------
        preprocessed_continuous_series_D : dictionary {number: np.array}
            The preprocessed and discretized continuous time series data.
        preprocessed_continuous_series_UD : dictionary {number: np.array}
            The preprocessed and UNdiscretized continuous time series data.
        preprocessed_event_logs : dictionary {number: np.array}
            The preprocessed and discretized discrete event logs.
        window_labels : np.array
            The windowed labels.

        Note: the time series are preprocessed separately.
        Note: each time series in the dictionary is written to a separate file.
        Note: the same preprocessing parameters are applied to each time series.
        Note: the event log is only divided in the appropriate windows.
        Note: MIN-MAX scaling is always applied!
        """

        # TODO: ignores multiple continous time series

        self._print_run_info()

        # TODO: this is some adhoc error handling
        # check the parameters
        if not isinstance(continuous_series, dict):
            sys.exit('ERROR preprocess: `continuous_series` parameter should be a dictionary containing the number + series')
        if not isinstance(event_logs, dict):
            sys.exit('ERROR preprocess: `event_logs` parameter should be a dictionary containing the number + event log')
        ts_lengths = []
        cs = False
        el = False
        for k, v in continuous_series.items():
            if not isinstance(k, int):
                raise Exception('ERROR preprocess: the continuous time series should be numbered (INT)')
            if not isinstance(v, np.ndarray):
                raise Exception('ERROR preprocess: continuous time series data should be numpy arrays')
            ts_lengths.append(len(v))
            cs = True
        for k, v in event_logs.items():
            if not isinstance(k, int):
                raise Exception('ERROR preprocess: the event logs should be numbered (INT)')
            if not isinstance(v, np.ndarray):
                raise Exception('ERROR preprocess: event log data should be numpy arrays')
            # event logs are assumed to be encoded as strings
            if v.dtype.kind not in ['U']:
                print('WARNING preprocess: the event log is expected to be an array of strings, in each string, events are separated by `-`')
                v = np.array([str(e) for e in v])
            event_logs[k] = v.astype(str)
            ts_lengths.append(len(v))
            el = True
        if len(ts_lengths) == 0:
            raise Exception('ERROR preprocess: no continuous time series or event logs provided in the required format')
        if len(set(ts_lengths)) != 1:
            raise Exception('ERROR preprocess: each time series should have the same length (event log placeholder is assumed to be the empty string)')
        if el:
            print('Event logs are preprocessed without: scaling, smoothing, discretizing. Only subsampling + binning + windowing are applied.')

        n = ts_lengths[0]

        if len(labels) > 0:
            if not isinstance(labels, np.ndarray):
                sys.exit('ERROR preprocess: labels should by passed as a numpy array')
            if len(labels) != n:
                sys.exit('ERROR preprocess: label array should have equal length to the time series, labels (-1, 0, 1)')
        lbl_cntr = Counter(labels)
        if lbl_cntr[1.0] > 2 * lbl_cntr[-1.0]:
            print('WARNING preprocess: more than twice the number of labeled anomalies as normals in `labels` (CAN BE POSSIBLE)')

        write_to_file = False
        if not isinstance(output_location, str):
            print('WARNING preprocess: invalid format for specifying `output_location`, not writing to file')
        else:
            if output_location != '':
                write_to_file = True
                if not(os.path.exists(output_location)):
                    os.makedirs(output_location)
        if self.verbose and write_to_file:
            print('Writing the output to file location: {}'.format(output_location))

        # each time series gets processed separately
        # preprocess the continuous time series
        preprocessed_continuous_series_D = {}
        preprocessed_continuous_series_UD = {}
        for ts_number, series in continuous_series.items():
            # remove extreme values
            if self.remove_extremes:
                mu = np.mean(series)
                std = np.std(series)
                def _extreme_func(e):
                    if e > mu + 3 * std:
                        return mu + 3 * std
                    elif e < mu - 3 * std:
                        return mu - 3 * std
                    else:
                        return e
                vec_extreme_func = np.vectorize(_extreme_func)
                series = vec_extreme_func(series)

            # min-max scaling
            if self.minmax_scaling:
                smin = np.min(series)
                smax = np.max(series)
                series = (series - smin) / (smax - smin)

            # additional scaling as specified by the user
            if self.add_scaling:
                def _scale_func(e):
                    return min(self.capvalue, e * self.scaler)
                vec_scale_func = np.vectorize(_scale_func)
                series = vec_scale_func(series)

            # smoothing (currently deals with edges suboptimally)
            if self.smoothing:
                w_left = max(1, int((self.smooth_incr - 1) / 2))
                w_right = max(1, int((self.smooth_incr + 1) / 2))
                for i in range(n):
                    series[i] = np.mean(series[max(0, i-w_left):min(n, i+w_right)])

            # subsampling
            if self.subsample > 1:
                series = series[::self.subsample]

            # binning
            if self.bin_size > 1:
                series_binned = np.array([np.mean(series[i*self.bin_size:(i+1)*self.bin_size]) for i in range(math.ceil(n/self.bin_size))])
            else:
                series_binned = series

            # discretizing
            if self.discretize:
                """ Two strategies for discretizing the data are possible here:
                    1. equal-density binning
                            Each bin contains an equal amount of observed values.
                    2. equal-width binning
                            Each bin has an equal widht, irrespective of the number
                            of observed values in the data.

                    Here, np.histogram() does equal-width binning of the data.
                """
                _, bin_edges = np.histogram(series_binned, bins=self.alphabet_size)
                discretized_alphabet_values = np.array([float('{0:0.2f}'.format(np.mean(bin_edges[i:i+1]))) for i in range(len(bin_edges))])
                value_ids = np.digitize(series_binned, bin_edges, right=False) - 1
                series_discrete = np.array([discretized_alphabet_values[i] for i in value_ids])
            else:
                series_discrete = series_binned

            # windowing
            discrete_windows = self._fast_divide_series_into_windows(series_discrete, data_type='continuous')
            binned_windows = self._fast_divide_series_into_windows(series_binned, data_type='continuous')

            preprocessed_continuous_series_D[ts_number] = discrete_windows
            preprocessed_continuous_series_UD[ts_number] = binned_windows

        # preprocess the event logs
        preprocessed_event_logs = {}
        for ts_number, series in event_logs.items():
            # subsampling
            if self.subsample > 1:
                series = series[::self.subsample]

            # windowing: if binning occurs, this adapts both the window size and window increment
            event_windows = self._fast_divide_series_into_windows(series, data_type='event')

            # encode the event log as ints
            event_windows = self._encode_events_as_int(event_windows)
            preprocessed_event_logs[ts_number] = event_windows

        # the labels are processed to match the windowing
        if len(labels) > 0:
            window_labels = self._fast_divide_labels_into_windows(labels)
        else:
            window_labels = np.zeros(n, dtype=np.float64)

        # return or write to file
        if write_to_file:
            print('ERROR preprocess: `writing to file` not implemented yet')

        # also return the series_binned, not only series_discrete (make features vs mine patterns)
        if return_undiscretized:
            return preprocessed_continuous_series_D, preprocessed_continuous_series_UD, preprocessed_event_logs, window_labels
        return preprocessed_continuous_series_D, preprocessed_event_logs, window_labels

    def _fast_divide_series_into_windows(self, series, data_type):
        """ Divide the time series into windows """
        n = len(series)

        if data_type == 'continuous':
            nw = math.ceil((n - self.window_size) / self.window_incr) + 1
            windowed_series = np.zeros((nw, self.window_size), dtype=np.float64)
            for i in range(nw):
                w = i * self.window_incr
                segment = series[w:w+self.window_size]
                windowed_series[i, :len(segment)] = segment
        elif data_type == 'event':
            # binning adapts window size and increment
            nw = math.ceil((n - (self.window_size * self.bin_size)) / (self.window_incr * self.bin_size)) + 1
            windowed_series = []
            for i in range(nw):
                w = i * (self.window_incr * self.bin_size)
                segment = series[w:w+self.window_size*self.bin_size]
                row = np.array('-'.join(segment).split('-'))
                row = np.array([r for r in row if not(r == '')])
                ix = sorted(np.unique(row, return_index=True)[1])
                windowed_series.append(row[ix])
            windowed_series = np.array(windowed_series)
        else:
            raise Exception('ERROR preprocess (_fast_divide_series_into_windows): unknown time series type')

        return windowed_series

    def _fast_divide_labels_into_windows(self, labels):
        """ Redistribute the labels so they match the constructed windows
            --> should consider: subsampling, binning, window size, window increment

        Label annotation scheme:
            1. lbl_segment contains no labels --> no label
            2. lbl_segment contains anomaly label --> anomaly
            3. lbl_segment contains no anomaly label and normal label --> normal

        Note: the subsampling should have no impact
        """
        n = len(labels)
        lws = int(self.window_size * self.bin_size * self.subsample)
        lwi = int(self.window_incr * self.bin_size * self.subsample)
        nw = math.ceil((n - lws) / lwi) + 1

        window_labels = np.zeros(nw, dtype=np.float64)
        for i in range(nw):
            lbl_segment = labels[i*lwi:i*lwi+lws]
            # label annotation scheme: multiple are possible
            if self.label_scheme == 1:
                # for the univariate data
                cntr = Counter(lbl_segment)
                if cntr[-1.0] == 0 and cntr[1.0] == 0:
                    continue
                elif cntr[1.0] >= 1:
                    window_labels[i] = 1.0
                else:
                    window_labels[i] = -1.0
            elif self.label_scheme == 2:
                # for the multivariate data
                cntr = Counter(lbl_segment)
                if cntr[1.0] > math.floor(lws / 2):
                    window_labels[i] = 1.0
                elif cntr[-1.0] > math.floor(lws / 2):
                    window_labels[i] = -1.0
                else:
                    pass
            else:
                raise Exception('ERROR: `label_scheme` with value ' + str(self.label_scheme) + ' not implemented yet!')

        return window_labels

    def _encode_events_as_int(self, event_log):
        """ Encode the event log data as integers """
        unique_strs = np.unique(np.concatenate(event_log))
        encode_dct = {u: i for i, u in enumerate(unique_strs) if not u == ''}  # ignore the empty string
        # TODO: vectorize
        encoded_log = []
        for w in event_log:
            new_w = [encode_dct[val] for val in w if not(val == '')]
            encoded_log.append(np.array(new_w))
        return encoded_log

    def _print_run_info(self):
        """ Print the settings of the preprocessor """
        if self.verbose:
            print('\nRunning preprocessor on TIME SERIES with settings & steps:')
            print('0. remove extreme values (mean +/- 3 * stdv)    {}'.format('YES' if self.remove_extremes else 'NO'))
            print('1. Min-Max scaling      {}'.format('YES' if self.minmax_scaling else 'NO'))
            print('2. Additional scaling:  {}'.format('YES' if self.add_scaling else 'NO'))
            if self.add_scaling:
                print('   scaler:              {}'.format(self.scaler))
                print('   capvalue:            {}'.format(self.capvalue))
            print('3. Smoothing:           {}'.format('YES: moving average' if self.smoothing else 'NO'))
            if self.smoothing:
                print('   smooth_incr:         {}'.format(self.smooth_incr))
            print('4. Binning:             {}'.format('YES: ' + str(self.bin_size) if self.bin_size > 1 else 'NO'))
            print('5. Subsampling:         {}'.format('YES: average per bin' + str(self.subsample) if self.subsample > 1 else 'NO'))
            print('6. Discretizing:        {}'.format('YES' if self.discretize else 'NO'))
            if self.discretize:
                print('   alphabet size:       {}'.format(self.alphabet_size))
            print('7. Window (size - inc): {} - {}'.format(self.window_size, self.window_incr))
        else:
            print('\nRunning preprocessor on TIME SERIES with settings')
