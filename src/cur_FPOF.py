import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from methods.PreProcessor import PreProcessor
from methods.pbad import PBAD
from baselines.FPOF import FPOF

from utils.casas_dataset import Casas

# 1. preprocess the data
casas = Casas("hh101")
data = casas.get_ann_raw_dataframe()
list_devices = data['Name'].unique()

new_data = data[data['Name'] == list_devices[1]]
'''
preprocesser = PreProcessor(window_size=12, window_incr=6, alphabet_size=30)
ts_windows_discretized, ts_windows, _, window_labels = preprocesser.preprocess(continuous_series=ts, labels=labels,
                                                                       return_undiscretized=True)

# run FPOF
detector = FPOF(relative_minsup=0.01, jaccard_threshold=0.9, pattern_pruning='closed')
scores = detector.fit_predict(ts_windows_discretized)

# 3. evaluation on labeled segments
filter_labels = np.where(window_labels != 0)[0]
print(len(window_labels),len(filter_labels), len(scores))
print('AUROC =', roc_auc_score(y_true=window_labels[filter_labels], y_score=scores[filter_labels]))
'''
