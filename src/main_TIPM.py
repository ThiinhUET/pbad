"""

Used for integration with TIPM: A tool voor Interactive time series 
pattern mining and anomaly detection. (https://bitbucket.org/len_feremans/tipm_pub).
For TIPM we run PBAD_Embed commansd-line, which is PBAD without preprocessing and pattern mining first,
since this is done by this tools. PBAD_Embed computes weighted occurences and an isolation forest
"""
import sys, os
import pandas as pd
import numpy as np
from methods.PBAD_Embed import PBAD_Embed
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict

#Convert nested list of windows to 2d numpy array
#Problem: if windows have different dimensions, np.array does not create matrix,
#but list of objects.
#Create matrix and pad windows with 0's if necessary
def windows2numpy(listOfWindows):
    normal_length = len(listOfWindows[len(listOfWindows)//2])
    listOfWindows2 = [];
    for i in range(0, len(listOfWindows)):
        lst1 = listOfWindows[i]
        lenLst1 = len(lst1) 
        if lenLst1 != normal_length:
            if lenLst1 > normal_length:
                raise Exception("Length is higher than expected")
            else:
                for i in range(0, normal_length - lenLst1):
                    lst1.append(0.0)
        for idx, val in enumerate(lst1): #bug in PBAD, called from TIPM, if empty values
            if val == '?':
                lst1[idx] = 0.0
        np_arr = np.array(lst1).astype(np.float64)
        listOfWindows2.append(np_arr)
    np_arr = np.array(listOfWindows2)
    print('Debug: windows2numpy: type {}, type(arr[0]) {}, type(arr[0][0]) {} shape {}, arr[0] {}'.format(
        type(np_arr),
        type(np_arr[0]), 
        type(np_arr[0][0]), 
        np_arr.shape,
        np_arr[i][0]))
    return np_arr

if __name__ == '__main__':
    #parse arguments
    usage = "main_TIPM -input CSVFILE  -type all -columns pc1,pc2\n" + \
            "-itemset_fnames pc1_closed_item.txt,pc2_closed_item.txt\n" + \
            "-sequential_fnames pc1_closed_sp.txt,pc2_closed_sp.txt\n" + \
            "-score_fname output.txt"
    arguments = sys.argv
    print('Argument List:' + str(arguments))
    if '-?' in arguments:
        print(usage)
        sys.exit(0) #normal end, for -? parameter
    if not('-type' in arguments and '-columns' in arguments and '-input' in arguments
        and ('-itemset_fnames' in arguments or '-sequential_fnames' in arguments)):
        print(usage)
        sys.exit(-1)
        
    def get_argument(key):
        for idx, arg in enumerate(arguments):
            if arg.strip().lower() == key:
                if idx != len(arguments)-1:
                    return arguments[idx+1]
                else:
                    raise Exception("Illegal last argument. " + str(arguments))
        return None
    inputfilename = get_argument('-input')
    pattern_type = get_argument('-type')
    columns = get_argument('-columns').lower().split(',')
    itemset_fnames = get_argument('-itemset_fnames')
    sequential_fnames = get_argument('-sequential_fnames')
    score_fname = get_argument('-score_fname')
    #Validation command-line arguments
    # 1) Type is either all, itemset, sequential
    # 2) Depending on type we expect either an file with either itemsets and/or sequential pattern for each column 
    if not pattern_type in ['all', 'itemset', 'sequential']:
        print('Type not in ' + str(['all', 'itemset', 'sequential'])); 
        print(usage)
        sys.exit(-1)
    if not os.path.isfile(inputfilename):
        print('input does not exist') 
        print(usage)
        sys.exit(-1)
    if (pattern_type == 'all' or pattern_type=='itemset') and itemset_fnames == None:
        print('Specify -itemset_fnames') 
        print(usage)
        sys.exit(-1)
    if (pattern_type == 'all' or pattern_type=='sequential') and sequential_fnames == None:
        print('Specify -sequential_fnames') 
        print(usage)
        sys.exit(-1)
    for fnames in [itemset_fnames, sequential_fnames]:
        if fnames != None:
            for idx, fname in enumerate(fnames.split(',')):
                if not os.path.isfile(fname):
                    print('pattern input does not exist ' + fname) 
                    print(usage)  
                    sys.exit(-1) 
                else:
                    f = open(fname, 'r')
                    l1 = f.readline().lower().split(',')
                    l2 = f.readline().lower().split(',')
                    print(str(idx) + ': Reading patterns ' + fname + ' for testing\n' + str(l1) + '\n' + str(l2))
                    #print('   Associate column: ' + columns[idx])
                    f.close()      
                    
    #Validation CSV file
    # Assumes CSV file has following structure:
    # 1) First column is timestamp/time step
    # 2) Label column is named "label"
    # 3) Window column is named "window"
    # 4) For each continous time series with name X, the corresponding columns has name X_D
    # 5) Patternset are 1 dimensional
    f = open(inputfilename, 'r')
    columns_csv = f.readline().lower().strip().split(',')
    f.close()     
    print('Reading CSVFile ' + str(columns_csv))
    if not 'window' in columns_csv:   
        print('Expecting column window')
        sys.exit(-1) 
    if not 'label' in columns_csv:   
        print('Expecting column label')
        sys.exit(-1)   
    #If discrete column names are pased, fix this
    columns = [col if not col.endswith('_d') else col[0:len(col)-2] for col in columns]  
    for col in columns:
        if not col in columns_csv:
            print('Expecting time series column ' + col)
            sys.exit(-1)   
        if not col + '_d' in columns_csv:
            print('Excepting time series discretized column with name ' + col + '_d')
            sys.exit(-1)     
   
    #RUN
    #preprocess: create windows for each continuous column, i.e. group by window column in TIPM
    #            for labels create either 1 (anomaly) if 1 is in window, or -1 (good) if -1 in window and not 1, else 0
    #Note: Doing this in plain-old python, instead of using more efficient numpy stuff
    df = pd.read_csv(inputfilename, header=0, index_col=0)
    cols = [c.lower().strip() for c in list(df.columns.values)]
    rows =  df.values.tolist()
    windowIdx = cols.index("window")
    labelIdx = cols.index("label")
    columnsIdx = [cols.index(col) for col in columns]
    discrete_columnsIdx = [cols.index(col+'_d') for col in columns]                         
    group_by_window = defaultdict(list)
    current_window = None
    windows = list()
    for row in rows:
        window = row[windowIdx]
        if not window in windows:
            windows.append(window)
        group_by_window[window].append(row)
    windowed_labels = []
    windowed_series = [[] for col in columnsIdx]
    windowed_series_discrete = [[] for col in discrete_columnsIdx]
    for win in windows:
        rows_matching_window = group_by_window[win]
        labels = []
        series = [[] for col in columnsIdx]
        discrete_series = [[] for col in discrete_columnsIdx]
        for row in rows_matching_window:
            labels.append(row[labelIdx])
            for i, colIdx in enumerate(columnsIdx):
                series[i].append(row[colIdx])
            for i, colIdx in enumerate(discrete_columnsIdx):
                discrete_series[i].append(row[colIdx])
        single_label = 0
        if -1 in labels:
            single_label = -1
        if 1 in labels:
            single_label = 1
        windowed_labels.append(single_label)
        for i in range(0, len(columnsIdx)):
            windowed_series[i].append(series[i])
        for i in range(0, len(discrete_columnsIdx)):
            windowed_series_discrete[i].append(discrete_series[i])
    #transform to datastructures for PBAD
    window_labels=np.array(windowed_labels)
    continuous_data = {}
    continuous_data_discretized={}
    for i in range(0, len(columnsIdx)):
        continuous_data[i] = windows2numpy(windowed_series[i])
        continuous_data_discretized[i] = windows2numpy(windowed_series_discrete[i])
    #cont_series = {0: data.iloc[:, 0].values}
    #labels = data.iloc[:, 1].values
    #cd_D, cd_UD, _, window_labels = preprocess(cont_series, labels=labels)       
    # run PBAD, sequential_fnames]:
    print('\nRunning PBAD Embed: This computes embedding of patterns, that is a weighted occurrences score for each pattern and each window,' + \
          'and than compute an anomaly score using isolation forests. Patternsets must be provided.')
    if itemset_fnames != None:
        itemset_fnames = itemset_fnames.split(',')
    if sequential_fnames != None:
        sequential_fnames = sequential_fnames.split(',')
    detector = PBAD_Embed(pattern_type=pattern_type, itemset_filenames_cont=itemset_fnames, sp_filenames_cont=sequential_fnames)
    scores = detector.fit_predict(continuous_data_discretized, continuous_data)
    ixl = np.where(window_labels != 0)[0]
    auc = roc_auc_score(y_true=window_labels[ixl], y_score=scores[ixl])
    ap = average_precision_score(y_true=window_labels[ixl], y_score=scores[ixl])
    print("AUC: {:.3f}".format(auc))
    print("AP: {:.3f}".format(ap))   
    #save score
    if score_fname != None:
        f = open(score_fname, 'w')
        f.write("Window,Score\n")    
        for idx, win in enumerate(windows):
            score = scores[idx]
            f.write("{},{:.6f}\n".format(win,score)) 
        f.close()         
        print("Saved {}".format(score_fname))