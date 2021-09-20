## PBAD: Pattern-Based Anomaly detection

Implementation of _Pattern-Based Anomaly Detection in Mixed-Type Time Series_, 
by Vincent Vercruyssen and Len Feremans. 

Paper published  at the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases 2019 (ECML-PKDD).
> The present-day accessibility of technology enables easy logging of both sensor values and event logs over extended periods. In this context, detecting abnormal segments in time series data has become an important data mining task. Existing work on anomaly detection focuses either on continuous time series or discrete event logs and not on the combination. However, in many practical applications, the patterns extracted from the event log can reveal contextual and operational conditions of a device that must be taken into account when predicting anomalies in the continuous time series. This paper proposes an anomaly detection method that can handle mixed-type time series. The method leverages frequent pattern mining techniques to construct an embedding of mixed-type time series on which an isolation forest is trained. Experiments on several real-world univariate and multivariate time series, as well as a synthetic mixed-type time series, show that our anomaly detection algorithm outperforms state-of-the-art anomaly detection techniques such as MatrixProfile, Pav, Mifpod and Fpof.

[Full paper](http://adrem.uantwerpen.be//bibrem/pubs/pbad.pdf)

An interactive tool for TIme series Pattern Mining (*TiPM*) that contains PBAD, is available [here](https://bitbucket.org/len_feremans/tipm_pub/).

### Summary

**PBAD** takes a *mixed-type time series* as input. A mixed-type timeserie consist of multiple continuous time series, together with one or more discrete event logs. **PBAD** computes an anomaly score for each window without the need for *labels*.

**PBAD** consist of 4 major steps:

1. Preprocessing univariate, multivariate, and mixed-type time series.
2. Mining a (non-redundant) set of *itemsets* and *sequential patterns* from each time series.
3. Constructing an *embedding* of all time series based on *distance-weighted pattern occurrences*.
4. Detecting anomalies using an *isolation forest*.
		
### Installation 

1. Clone the repository
2. Code is implemented in `Python`, but some performance-critical code is implemented in `C` using `Cython`. Build the Cython code by running the setup.py file:

```bash
cd src/utils/cython_utils/

python setup.py build_ext --inplace
```

### Usage
**PBAD** consists of `methods.PreProcessor` for pre-processing and `methods.PBAD` for predicting contextual anomalies.

Parameters for `methods.PreProcessor` are:

- `window_size` and `window_incr(ement)` control the creation of _fixed sized sliding windows_ in discrete and continous time series.
- `bin_size` can be used for downsampling a continuous timeseries using a _moving average_.
- `alphabet_size` controls the number of bins for _equal-width discretisation_.

Parameters for `methods.PBAD` are:

- `relative_minsup` controls the amount of itemsets and sequential pattern generated (default is 0.01)
- `jaccard_threshold` controls the filtering of redundant patterns (between 0.0 and 1.0) 
- `pattern_pruning` is either `maximal` or `closed` (default is `maximal`) 
- `pattern_type` is either `itemset`, `sequential` or `all` (meaning both types of patterns, default is `all`)

We illustrate both classes in the following example:

```python
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from methods.PreProcessor import PreProcessor
from methods.PBAD import PBAD

# Univariate input file has three columns: timestamp, value and label.
# Label is either 0=unknown, 1=normal or -1=abnormal
# timestamp,value,label
# 2013-07-04 00:00:00,0.43,0
# 2013-07-04 01:00:00,0.48,0
input_file =  './univariate/ambient_temperature/train_data.csv'

# 1. preprocess the data
univariate_data = pd.read_csv(input_file, header=0, index_col=0) #index on timestamp column
ts = {0: univariate_data.iloc[:, 0].values} #value column
labels = univariate_data.iloc[:, 1].values  #label column

preprocesser = PreProcessor(window_size=12, window_incr=6, alphabet_size=30)
ts_windows_discretized, ts_windows, _, window_labels = preprocesser.preprocess(continuous_series=ts, labels=labels,
                                                                       return_undiscretized=True)

# 2. run PBAD on the data
pbad = PBAD(relative_minsup=0.01, jaccard_threshold=0.9, pattern_type='all', pattern_pruning='maximal')
scores = pbad.fit_predict(ts_windows, ts_windows_discretized)

# 3. evaluation on labeled segments
filter_labels = np.where(window_labels != 0)[0]
print('AUROC =', roc_auc_score(y_true=window_labels[filter_labels], y_score=scores[filter_labels])) #AUROC = 0.997
```

### More information for researchers and contributors ###
The current version is 0.9, and was last update on september 2019. The main implementation is written in `Python`.  Performance-critical code, mainly for computing the embeding based on weighted pattern-based occurrences, is implemented in `C` using `Cython`.  For mining closed, maximal and minimal infrequent itemsets and sequential patterns we depend on the `Java`-based [SPMF](www.philippe-fournier-viger.com/spmf/) library. Python Dependencies are `numpy==1.16.3`, `pandas==0.24.2`, `scikit-learn==0.20.3`, `Cython==0.29.7` and `scipy==1.2.1`.
 
We compare performance with the following state-of-the-art methods, which we implemented:

- `FPOF`: [Fp-outlier: Frequent pattern based outlier detection](https://www.researchgate.net/profile/Zengyou_He/publication/220117736_FP-outlier_Frequent_pattern_based_outlier_detection/links/53d9dec60cf2e38c63363c05/FP-outlier-Frequent-pattern-based-outlier-detection.pdf). 
- `PAV`: [Multi-scale anomaly detection algorithm based on infrequent pattern of time series](https://core.ac.uk/download/pdf/81941819.pdf).
- `MIFPOD`: [Minimal infrequent pattern based approach for mining outliers in data streams](https://www.sciencedirect.com/science/article/pii/S0957417414006149).
- `MP`: [Matrix profile, all pairs similarity joins for time series: a unifying view that includes motifs, discords and shapelets](https://ieeexplore.ieee.org/document/7837992]) for anomaly detection. 

Datasets are provided in _/data_:

- `univariate` *New york taxi*, *ambient temperature*, and *request latency*. Origin is the [Numenta repository](https://github.com/numenta).
- `multivariate` *Indoor physical exercises* dataset captured using a Microsoft Kinect camera. Origin is [AMIE: Automatic Monitoring of Indoor Exercises](https://dtai.cs.kuleuven.be/software/amie).
- `mixed-type` *Synthetic power grid* dataset. See `experiments`.`synthetic_mixed_type_data_generator` for details.

To run experiments that compare **PBAD** with state-of-the-art methods run `experiments`.`reproduce_experiments`. 
   
### Contributors

- Vincent Vercruyssen,  DTAI research group, University of Leuven, Belgium.
- Len Feremans, Adrem Data Lab research group, University of Antwerp, Belgium.

### Licence ###
Copyright (c) [2019] [Len Feremans and Vincent Vercruyssen]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFWARE.