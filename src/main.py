""" Run test.
"""

import sys, os, time
import numpy as np
import pandas as pd
import scipy as sp

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from methods.PreProcessor import PreProcessor
from methods.PBAD import PBAD
from baselines.FPOF import FPOF
from baselines.PAV import PAV
from baselines.MPAD import MPAD
from baselines.MIFPOD import MIFPOD

#from utils.pattern_mining import mine_non_redundant_itemsets, mine_non_redundant_sequential_patterns
from utils.make_features import make_pattern_based_features
from utils.pattern_mining import mine_non_redundant_itemsets, mine_non_redundant_sequential_patterns, mine_minimal_infrequent_itemsets


def main():
    print('imports succesful')

if __name__ == '__main__':
    sys.exit(main())
