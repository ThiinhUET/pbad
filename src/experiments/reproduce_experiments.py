"""
pattern-based anomaly detection
-------------------------------
Reproduce the experimental results in the paper.

:authors: Vincent Vercruyssen & Len Feremans
:copyright:
    Copyright 2019 KU Leuven, DTAI Research Group.
    Copyright 2019 UAntwerpen, ADReM Data Lab.
:license:

"""

import sys, os, time, math
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import roc_auc_score, average_precision_score

try:
    script_loc, _ = os.path.split(os.path.realpath(__file__))
    SRC_LOC = os.path.split(script_loc)[0]
    sys.path.insert(0, SRC_LOC)
except Exception as e:
    print(e)
    sys.exit()

from methods.PreProcessor import PreProcessor
from methods.PBAD import PBAD
from baselines.FPOF import FPOF
from baselines.MPAD import MPAD
from baselines.MIFPOD import MIFPOD
from baselines.PAV import PAV
    

################################################################################
# REPRODUCE EXPERIMENTAL RESULTS: 
################################################################################
def main():
    print('Starting experiments')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path.split('src/experiments')[0], 'data')

    # 1. reproduce the univariate experiments
    print('\nReproducing UNIVARIATE:')
    #reproduce_univariate_experiments(data_path)

    # 2. reproduce the multivariate experiments
    print('\nREPRODUCING MULTIVARIATE:')
    reproduce_multivariate_experiments(data_path)


def reproduce_univariate_experiments(data_path):
    datasets = ['ambient_temperature', 'request_latency', 'new_york_taxi']
    auc_results = pd.DataFrame(0.0, columns=['MPAD', 'PAV', 'MIFPOD', 'FPOF', 'PBAD'], index=datasets)
    ap_results = pd.DataFrame(0.0, columns=['MPAD', 'PAV', 'MIFPOD', 'FPOF', 'PBAD'], index=datasets)
    
    for dataset in datasets:
        data = pd.read_csv(os.path.join(data_path, 'univariate', dataset, 'train_data.csv'), header=0, index_col=0)
        data_settings = pickle.load(open(os.path.join(data_path, 'univariate', dataset, 'data_settings.pickle'), 'rb'))
        cont_series = {0: data.iloc[:, 0].values}
        labels = data.iloc[:, 1].values

        # preprocess
        prep = PreProcessor(
            remove_extremes=True,
            minmax_scaling=False,
            add_scaling=False,
            smoothing=False,
            label_scheme=1,
            discretize=False,
            alphabet_size=int(data_settings['alphabet_size']),
            window_size=int(data_settings['wsize']),
            window_incr=int(data_settings['wincrement']),
            bin_size=int(data_settings['bin_size']))
        cd_D, cd_UD, _, window_labels = prep.preprocess(cont_series, labels=labels, return_undiscretized=True)
        ixl = np.where(window_labels != 0)[0]

        # run PBAD
        print('\nRunning PBAD:')
        detector = PBAD(relative_minsup=0.01, jaccard_threshold=0.9, pattern_type='all', pattern_pruning='maximal', distance_formula=1, sequential_minlength=1.0)
        scores = detector.fit_predict(cd_UD, cd_D)
        auc = roc_auc_score(y_true=window_labels[ixl], y_score=scores[ixl])
        auc_results.loc[dataset, 'PBAD'] = auc
        ap = average_precision_score(y_true=window_labels[ixl], y_score=scores[ixl])
        ap_results.loc[dataset, 'PBAD'] = ap

        # run FPOF
        print('\nRunning FPOF (settings based on best experiments for this method):')
        if dataset == 'ambient_temperature':
            detector = FPOF(relative_minsup=0.01, jaccard_threshold=0.9, pattern_pruning='closed')
        elif dataset == 'new_york_taxi':
            detector = FPOF(relative_minsup=0.05, jaccard_threshold=0.9, pattern_pruning='closed')
        else:
            detector = FPOF(relative_minsup=0.1, jaccard_threshold=0.9, pattern_pruning='closed')
        scores = detector.fit_predict(cd_D)
        auc = roc_auc_score(y_true=window_labels[ixl], y_score=scores[ixl])
        auc_results.loc[dataset, 'FPOF'] = auc
        ap = average_precision_score(y_true=window_labels[ixl], y_score=scores[ixl])
        ap_results.loc[dataset, 'FPOF'] = ap

        # run PAV
        print('\nRunning PAV:')
        detector = PAV()
        scores = detector.fit_predict(cd_UD, window_size=12, window_incr=6)
        auc = roc_auc_score(y_true=window_labels[ixl], y_score=scores[ixl])
        auc_results.loc[dataset, 'PAV'] = auc
        ap = average_precision_score(y_true=window_labels[ixl], y_score=scores[ixl])
        ap_results.loc[dataset, 'PAV'] = ap

        # run MIFPOD
        print('\nRunning MIFPOD (settings based on best experiments for this method):')
        if dataset == 'new_york_taxi':
            detector = MIFPOD(relative_minsup=0.1)
        else:
            detector = MIFPOD(relative_minsup=0.05)
        scores = detector.fit_predict(cd_D)
        auc = roc_auc_score(y_true=window_labels[ixl], y_score=scores[ixl])
        auc_results.loc[dataset, 'MIFPOD'] = auc
        ap = average_precision_score(y_true=window_labels[ixl], y_score=scores[ixl])
        ap_results.loc[dataset, 'MIFPOD'] = ap

        # run MPAD
        print('\nRunning MPAD (settings based on best experiments for this method):')
        detector = MPAD(window_size=100)  # int(data_settings['wsize']) (mistake)
        scores = detector.fit_predict({0: data.iloc[:, 0].values})
        w_scores = prep._fast_divide_series_into_windows(scores, 'continuous')
        scores = np.sum(w_scores, axis=1).T
        auc = roc_auc_score(y_true=window_labels[ixl], y_score=scores[ixl])
        auc_results.loc[dataset, 'MPAD'] = auc
        ap = average_precision_score(y_true=window_labels[ixl], y_score=scores[ixl])
        ap_results.loc[dataset, 'MPAD'] = ap


    # print the results
    print('\n\n\n------------ AUC RESULTS -------------\n')
    print(auc_results)
    print('\n------------ AP RESULTS -------------\n')
    print(ap_results)


def reproduce_multivariate_experiments(data_path):
    datasets = ['lunges_vs_squats', 'lunges_and_sidelunges_vs_squats', 'sidelunges_vs_lunges', 'squats_vs_sidelunges']
    auc_results = pd.DataFrame(0.0, columns=['MPAD', 'PAV', 'MIFPOD', 'FPOF', 'PBAD'], index=datasets)
    ap_results = pd.DataFrame(0.0, columns=['MPAD', 'PAV', 'MIFPOD', 'FPOF', 'PBAD'], index=datasets)
    
    for dataset in datasets:
        data = data = pd.read_csv(os.path.join(data_path, 'multivariate', dataset, 'train_data.csv'), sep=',', header=0, index_col=[0])
        data_settings = pickle.load(open(os.path.join(data_path, 'multivariate', dataset, 'data_settings.pickle'), 'rb'))
        cont_series = {i: data.iloc[:, i].values for i in range(data.shape[1] - 1)}
        labels = data.iloc[:, -1].values

        # preprocess
        prep = PreProcessor(
            remove_extremes=True,
            minmax_scaling=False,
            add_scaling=False,
            smoothing=False,
            label_scheme=2,
            discretize=bool(data_settings['discretize']),
            alphabet_size=int(data_settings['alphabet_size']),
            window_size=int(data_settings['wsize']),
            window_incr=int(data_settings['wincrement']),
            bin_size=int(data_settings['bin_size']))
        cd_D, cd_UD, _, window_labels = prep.preprocess(cont_series, labels=labels, return_undiscretized=True)
        ixl = np.where(window_labels != 0)[0]

        # run PBAD
        print('\nRunning PBAD (should rerun multiple times and average results!):')
        detector = PBAD(relative_minsup=0.01, jaccard_threshold=0.9, pattern_type='all', pattern_pruning='closed', distance_formula=1, sequential_minlength=2.0)
        scores = detector.fit_predict(cd_UD, cd_D)
        auc = roc_auc_score(y_true=window_labels[ixl], y_score=scores[ixl])
        auc_results.loc[dataset, 'PBAD'] = auc
        ap = average_precision_score(y_true=window_labels[ixl], y_score=scores[ixl])
        ap_results.loc[dataset, 'PBAD'] = ap

        # run FPOF
        print('\nRunning FPOF (settings based on best experiments for this method):')
        detector = FPOF(relative_minsup=0.1, jaccard_threshold=0.9, pattern_pruning='closed')
        scores = detector.fit_predict(cd_D)
        auc = roc_auc_score(y_true=window_labels[ixl], y_score=scores[ixl])
        auc_results.loc[dataset, 'FPOF'] = auc
        ap = average_precision_score(y_true=window_labels[ixl], y_score=scores[ixl])
        ap_results.loc[dataset, 'FPOF'] = ap

        # run PAV
        print('\nRunning PAV:')
        detector = PAV()
        scores = detector.fit_predict(cd_UD, window_size=int(data_settings['wsize']), window_incr=int(data_settings['wincrement']))
        auc = roc_auc_score(y_true=window_labels[ixl], y_score=scores[ixl])
        auc_results.loc[dataset, 'PAV'] = auc
        ap = average_precision_score(y_true=window_labels[ixl], y_score=scores[ixl])
        ap_results.loc[dataset, 'PAV'] = ap

        # run MIFPOD
        print('\nRunning MIFPOD (settings based on best experiments for this method):')
        if dataset == 'squats_vs_sidelunges':
            detector = MIFPOD(relative_minsup=0.05)
        else:
            detector = MIFPOD(relative_minsup=0.1)
        scores = detector.fit_predict(cd_D)
        auc = roc_auc_score(y_true=window_labels[ixl], y_score=scores[ixl])
        auc_results.loc[dataset, 'MIFPOD'] = auc
        ap = average_precision_score(y_true=window_labels[ixl], y_score=scores[ixl])
        ap_results.loc[dataset, 'MIFPOD'] = ap

        # run MPAD
        print('\nRunning MPAD (settings based on best experiments for this method):')
        detector = MPAD(window_size=100)  # int(data_settings['wsize']) (mistake)
        scores = detector.fit_predict(cont_series)
        w_scores = prep._fast_divide_series_into_windows(scores, 'continuous')
        scores = np.sum(w_scores, axis=1).T
        auc = roc_auc_score(y_true=window_labels[ixl], y_score=scores[ixl])
        auc_results.loc[dataset, 'MPAD'] = auc
        ap = average_precision_score(y_true=window_labels[ixl], y_score=scores[ixl])
        ap_results.loc[dataset, 'MPAD'] = ap


    # print the results
    print('\n\n\n------------ AUC RESULTS -------------\n')
    print(auc_results)
    print('\n------------ AP RESULTS -------------\n')
    print(ap_results)


if __name__ == '__main__':
    sys.exit(main())
