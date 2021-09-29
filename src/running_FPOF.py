import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from src.methods.PreProcessor import PreProcessor
from src.methods.PBAD import PBAD
from src.baselines.FPOF import FPOF

# Univariate input file has three columns: timestamp, value and label.
# Label is either 0=unknown, 1=normal or -1=abnormal
# timestamp,value,label
# 2013-07-04 00:00:00,0.43,0
# 2013-07-04 01:00:00,0.48,0
input_file =  os.path.join(os.path.dirname(__file__), '../data/univariate/ambient_temperature/train_data.csv')

# 1. preprocess the data
univariate_data = pd.read_csv(input_file, header=0, index_col=0) #index on timestamp column
ts = {0: univariate_data.iloc[:, 0].values} #value column
labels = univariate_data.iloc[:, 1].values  #label column

preprocesser = PreProcessor(window_size=12, window_incr=6, alphabet_size=30)
ts_windows_discretized, ts_windows, _, window_labels = preprocesser.preprocess(continuous_series=ts, labels=labels,
                                                                       return_undiscretized=True)
'''
# 2. run PBAD on the data
pbad = PBAD(relative_minsup=0.01, jaccard_threshold=0.9, pattern_type='all', pattern_pruning='maximal')
scores = pbad.fit_predict(ts_windows, ts_windows_discretized)
'''

# run FPOF
detector = FPOF(relative_minsup=0.01, jaccard_threshold=0.9, pattern_pruning='closed')
scores = detector.fit_predict(ts_windows_discretized)

# 3. evaluation on labeled segments
filter_labels = np.where(window_labels != 0)[0]
print('AUROC =', roc_auc_score(y_true=window_labels[filter_labels], y_score=scores[filter_labels]))