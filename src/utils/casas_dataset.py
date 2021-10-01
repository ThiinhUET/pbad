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
    def __init__(self, name):
        self.name = name

    def get_ann_features(self):
        data = pd.read_csv(os.path.join(prefix,self.name, f"{self.name}.ann.features.csv"), header=0, index_col=0)
        return data
    
    def get_ann_raw_dataframe(self):
        self._standardlize_casas()
        data = pd.read_csv(os.path.join(prefix,self.name, f"{self.name}.ann.standarlize.txt"), delimiter = '\t', names=["Date","Time", "Name", "Position1", "Position2", "Message", "Controller", "Activity"])
        return data
    '''
    standarlize deliminator of cansas dataset
    '''
    def _standardlize_casas(self):
        with open(os.path.join(prefix, self.name, f"{self.name}.ann.txt")) as fin, open(os.path.join(prefix, self.name, f"{self.name}.ann.standarlize.txt"), 'w') as fout:
            for line in fin:
                fout.write("\t".join(line.split()) + "\n")
