"""
Code to run an experiment
"""

import sys, os, time, random
import numpy as np
import pandas as pd
import argparse
import pickle

from sklearn.metrics import roc_auc_score, average_precision_score

# add the path to the source folder
try:
    script_loc, _ = os.path.split(os.path.realpath(__file__))
    SRC_LOC = os.path.split(script_loc)[0]
    sys.path.insert(0, SRC_LOC)
except Exception as e:
    print(e)
    sys.exit()

from src.methods.PreProcessor import PreProcessor
from src.methods.PBAD import PBAD
from src.baselines.FPOF import FPOF
from src.baselines.MPAD import MPAD
from src.baselines.MIFPOD import MIFPOD
from src.baselines.PAV import PAV

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main():

    # parse the input arguments
    parser = argparse.ArgumentParser(description='Run pattern based anomaly detection')
    parser.add_argument('-d', type=str, default='', help='datafile directory')
    parser.add_argument('-r', type=str, default='', help='results directory')
    parser.add_argument('-rn', type=int, default=42, help='random seed for reproducing experiments')
    # remainder are the AD arguments
    args, unknownargs = parser.parse_known_args()
    random.seed(args.rn)

    print('\n\nStarting...')

    # additional settings 
    prep_settings, algo_settings = _parse_additional_settings(unknownargs)

    # load the data preprocess
    data, data_settings = _load_and_preprocess_data(args.d)
    cont_series = {i: data.iloc[:, i].values for i in range(data.shape[1] - 1)}
    labels = data.iloc[:, -1].values

    # preprocess the data
    """ DISCRETIZE THE NUMENTA DATA """
    make_discrete = bool(data_settings['discretize'])
    prep = PreProcessor(
        remove_extremes=True,
        minmax_scaling=False,
        add_scaling=False,                                  # data_settings['scaling']
        smoothing=False,
        label_scheme=int(prep_settings['label_scheme']),
        discretize=make_discrete,
        alphabet_size=int(data_settings['alphabet_size']),
        window_size=int(data_settings['wsize']),
        window_incr=int(data_settings['wincrement']),
        bin_size=int(data_settings['bin_size'])
    )
    cd_D, cd_UD, _, window_labels = prep.preprocess(cont_series, labels=labels, return_undiscretized=True)
    ixl = np.where(window_labels != 0)[0]

    # run the anomaly detector
    method = algo_settings['name']
    del algo_settings['name']
    if method == 'PBAD':
        detector = PBAD(**algo_settings)
        scores = detector.fit_predict(cd_UD, cd_D)  # undiscretized, discretized
    elif method == 'FPOF':
        detector = FPOF(**algo_settings)
        scores = detector.fit_predict(cd_D)
    elif method == 'PAV':
        detector = PAV(**algo_settings)
        scores = detector.fit_predict(cd_UD, window_size=data_settings['wsize'],
            window_incr=data_settings['wincrement'])
    elif method == 'MIFPOD':
        detector = MIFPOD(**algo_settings)
        scores = detector.fit_predict(cd_D)
    elif method == 'MPAD':
        detector = MPAD(**algo_settings)
        scores = detector.fit_predict(cont_series)
        w_scores = prep._fast_divide_series_into_windows(scores, 'continuous')
        scores = np.sum(w_scores, axis=1).T
    else:
        raise Exception('INPUT ERROR - unknown anomaly detection method')

    #print('AUC=', roc_auc_score(y_true=window_labels[ixl], y_score=scores[ixl]))
    #print('AP=', average_precision_score(y_true=window_labels[ixl], y_score=scores[ixl]))

    # store the results
    if not os.path.exists(args.r):
        os.makedirs(args.r)
    experiment_predictions = {'true': window_labels[ixl], 'prob': scores[ixl]}
    with open(os.path.join(args.r, 'experiment_predictions.pickle'), 'wb') as handle:
        pickle.dump(experiment_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # store the settings
    store_settings = {'AD_method_name': method}
    for k, v in data_settings.items():
        store_settings[k] = v
    for k, v in algo_settings.items():
        store_settings['AD_' + k] = v
    with open(os.path.join(args.r, 'experiment_settings.pickle'), 'wb') as handle:
        pickle.dump(store_settings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('\nDone!')


def _parse_additional_settings(unknownargs):
    """ Parse the preprocessing + algorithm settings defined by the user """

    # translate the unknownargs to additional settings
    additional_settings = {}
    if not(len(unknownargs) % 2 == 0):
        raise Exception('ERROR: the additional arguments should always come in pairs (key-value)!')
    n_args = int(len(unknownargs) / 2)
    for i in range(n_args):
        k = unknownargs[2*i].strip('-')
        v = unknownargs[2*i+1]
        additional_settings[k] = v

    # derive the preprocessing and method settings
    prep_settings = {}
    algo_settings = {}
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    for k, v in additional_settings.items():
        if is_number(v):
            v = float(v)
        if v == 'True':
            v = True
        if v == 'False':
            v = False
        if 'prep_' in k:
            prep_settings[k.split('prep_')[1]] = v
        elif 'algo_' in k:
            algo_settings[k.split('algo_')[1]] = v
        else:
            raise Exception('INPUT ERROR - unknown setting')
    
    return prep_settings, algo_settings


def _load_and_preprocess_data(directory):
    """ Load the data settings and data from the file directory """
    # load the settings
    data_settings = pickle.load(open(os.path.join(directory, 'data_settings.pickle'), 'rb'))

    # load the data
    if data_settings['data_type'] == 'univariate':
        data = pd.read_csv(os.path.join(directory, 'train_data.csv'), sep=',', header=0, names=['pc1', 'label'], usecols=[1, 2])
    elif data_settings['data_type'] == 'multivariate':
        data = pd.read_csv(os.path.join(directory, 'train_data.csv'), sep=',', header=0, index_col=[0])
    else:
        data = pd.read_csv(os.path.join(directory, 'train_data.csv'), sep=',', header=0, names=['var', 'events', 'label'], index_col=[0])

    # return the data
    return data, data_settings


if __name__ == '__main__':
    sys.exit(main())
