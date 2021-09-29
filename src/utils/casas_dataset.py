'''
#Declare Casas Dataset path on local machine
folder inclue:
    + *.ann.features.csv
    + *.ann.features.README.txt
    + *.ann.txt
    + *.rawdata.txt
    image of floor map
'''

import pandas as pd
import os
prefix = '/Users/admin/Documents/Casas Dataset/'

class Casas:
    def get_ann_features(name):
        data = pd.read_csv(os.path.join(prefix,name, f"{name}.ann.features.csv"), header=0, index_col=0)
        return data