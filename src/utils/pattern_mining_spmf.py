"""
Mine itemsets or sequential patterns using SPMF.

Special is the database representation:
    - We assume a set of windows of fixed length, each containing floating point values
    - We assume float point values are enumerable, that is the set of distinct
      floating point values is small, and can be represented by items using one-hot-encoding
    - Output patterns are either itemsets or sequential-patterns but not of integers,
      but round floating point values
"""

import os, sys, subprocess, time, random
from tempfile import NamedTemporaryFile
import numpy as np
import uuid, shutil
import itertools
from datetime import datetime
from collections import defaultdict

# path to the SPMF library
try:
    SPMF_PATH = os.path.join(os.path.dirname(__file__), '../lib/spmf.jar')
    if not(os.path.exists(SPMF_PATH)):
        raise Exception('ERROR: SPMF library not found. Expecting SPMF qt (see patter_mining.py): ' + SPMF_PATH)
except:
    sys.exit('ERROR: The SPMF pattern mining library cannot be found (wrong path)')


################################################################################
# PATTERN MINING
################################################################################

def mine_closed_itemsets(data, relative_minsup, TEMP_DIR):
    """ Mine the closed itemsets """
    print('\nDEBUG: Mining CLOSED ITEMSETS with SPMF CHARM; #rows:{} minsup relative: {}\n'.format(len(data), relative_minsup))
    input_file_spmf = _data_to_spmf_file_format_transaction_db(data, TEMP_DIR)
    output_file_spmf = NamedTemporaryFile(mode='w', dir=TEMP_DIR, delete=False, suffix='spmf_charm_output.txt')
    status = _run_spmf_algorithm('Charm_bitset', input_file_spmf, output_file_spmf.name, str(relative_minsup*100) + '%')
    patterns, supports = _spmf_output_to_patterns(output_file_spmf.name, pattern_type='itemset')
    # necessary to unlink!
    os.unlink(input_file_spmf)
    os.unlink(output_file_spmf.name)
    # return the patterns
    return patterns, supports

#Len: new 25/04/2019
def mine_rare_itemsets(data, relative_minsup, TEMP_DIR):
    """ Mine rare or minimal infrequent itemsets """
    print('\nDEBUG: Mining RARE ITEMSETS with SPMF AprioriRare; #rows:{} minsup relative: {}\n'.format(len(data), relative_minsup))
    input_file_spmf = _data_to_spmf_file_format_transaction_db(data, TEMP_DIR)
    output_file_spmf = NamedTemporaryFile(mode='w', dir=TEMP_DIR, delete=False, suffix='spmf_rare_output.txt')
    status = _run_spmf_algorithm('AprioriRare', input_file_spmf, output_file_spmf.name, str(relative_minsup*100) + '%')
    patterns, supports = _spmf_output_to_patterns(output_file_spmf.name, pattern_type='itemset')
    # necessary to unlink!
    os.unlink(input_file_spmf)
    os.unlink(output_file_spmf.name)
    # return the patterns
    return patterns, supports

def mine_maximal_itemsets(data, relative_minsup, TEMP_DIR):
    """ Mine the maximal itemsets """
    print('\nDEBUG: Mining MAXIMAL ITEMSETS with CHARM MFI; #rows:{} minsup relative: {}\n'.format(len(data), relative_minsup))
    input_file_spmf = _data_to_spmf_file_format_transaction_db(data, TEMP_DIR)
    output_file_spmf = NamedTemporaryFile(mode='w', dir=TEMP_DIR, delete=False, suffix='spmf_charm_mfi_output.txt')
    status = _run_spmf_algorithm('Charm_MFI', input_file_spmf, output_file_spmf.name, str(relative_minsup*100) + '%')
    patterns, supports = _spmf_output_to_patterns(output_file_spmf.name, pattern_type='itemset')
    # necessary to unlink!
    os.unlink(input_file_spmf)
    os.unlink(output_file_spmf.name)
    # return the patterns
    return patterns, supports

#Len: new 25/04/2019
#Remark: CM-ClaSP can not handle REPEATING items in sequential patterns! See test_pattern_mining_spmf
def mine_closed_sequential_patterns(data, relative_minsup, TEMP_DIR):
    """ Mine the closed sequential patterns """
    print('\nDEBUG: Mining CLOSED SEQUENTIAL PATTERNS with ClasP; #rows:{} minsup relative: {}\n'.format(len(data), relative_minsup))
    input_file_spmf = _data_to_spmf_file_format_sequence_db(data, TEMP_DIR)
    output_file_spmf = NamedTemporaryFile(mode='w', dir=TEMP_DIR, delete=False, suffix='spmf_clasp_output.txt')
    status = _run_spmf_algorithm('ClaSP', input_file_spmf, output_file_spmf.name, str(relative_minsup*100) + '%')
    patterns, supports = _spmf_output_to_patterns(output_file_spmf.name, pattern_type='sequential_pattern')
    # necessary to unlink!
    os.unlink(input_file_spmf)
    os.unlink(output_file_spmf.name)
    # return the patterns
    return patterns, supports

# old: CS-ClaSP
def mine_closed_sequential_patterns_CM_CLASP(data, relative_minsup, TEMP_DIR):
    """ Mine the closed sequential patterns """
    print('\nDEBUG: Mining CLOSED SEQUENTIAL PATTERNS with CM-ClaSP; #rows:{} minsup relative: {}\n'.format(len(data), relative_minsup))
    input_file_spmf = _data_to_spmf_file_format_sequence_db(data, TEMP_DIR)
    output_file_spmf = NamedTemporaryFile(mode='w', dir=TEMP_DIR, delete=False, suffix='spmf_spade_output.txt')
    status = _run_spmf_algorithm('CM-ClaSP', input_file_spmf, output_file_spmf.name, str(relative_minsup*100) + '%')
    patterns, supports = _spmf_output_to_patterns(output_file_spmf.name, pattern_type='sequential_pattern')
    # necessary to unlink!
    os.unlink(input_file_spmf)
    os.unlink(output_file_spmf.name)
    # return the patterns
    return patterns, supports

#Vincent: new 27/06/2019
#Faster method for mining maximal sequential patterns
def mine_maximal_sequential_patterns_VMSP(data, relative_minsup, TEMP_DIR):
    """ Mine the maximal sequential patterns """
    print('\nDEBUG: Mining MAXIMAL SEQUENTIAL PATTERN with VMSP; #rows:{} minsup relative: {}\n'.format(len(data), relative_minsup))
    input_file_spmf = _data_to_spmf_file_format_sequence_db(data, TEMP_DIR)
    output_file_spmf = NamedTemporaryFile(mode='w', dir=TEMP_DIR, delete=False, suffix='spmf_maxsp_output.txt')
    status = _run_spmf_algorithm('VMSP', input_file_spmf, output_file_spmf.name, str(relative_minsup*100) + '%')
    patterns, supports = _spmf_output_to_patterns(output_file_spmf.name, pattern_type='sequential_pattern')
    # necessary to unlink!
    os.unlink(input_file_spmf)
    os.unlink(output_file_spmf.name)
    # return the patterns
    return patterns, supports

# old: MaxSP
def mine_maximal_sequential_patterns(data, relative_minsup, TEMP_DIR):
    """ Mine the maximal sequential patterns """
    print('\nDEBUG: Mining MAXIMAL SEQUENTIAL PATTERN with MaxSP; #rows:{} minsup relative: {}\n'.format(len(data), relative_minsup))
    input_file_spmf = _data_to_spmf_file_format_sequence_db(data, TEMP_DIR)
    output_file_spmf = NamedTemporaryFile(mode='w', dir=TEMP_DIR, delete=False, suffix='spmf_maxsp_output.txt')
    status = _run_spmf_algorithm('MaxSP', input_file_spmf, output_file_spmf.name, str(relative_minsup*100) + '%')
    patterns, supports = _spmf_output_to_patterns(output_file_spmf.name, pattern_type='sequential_pattern')
    # necessary to unlink!
    os.unlink(input_file_spmf)
    os.unlink(output_file_spmf.name)
    # return the patterns
    return patterns, supports


################################################################################
# TOP LEVEL HELPER FUNCTIONS
################################################################################

def _run_spmf_algorithm(algorithm, *args):
    """ Run the algorithm in SPMF (JAVA).
        Default timeout is 6 hours.
    """
    command = ['java', '-jar', SPMF_PATH, 'run', algorithm]
    for arg in args:
        command.append(str(arg))
    try:
        return subprocess.call(command, timeout=1000*60*6)
    except Exception as e:
        sys.exit('ERROR: could not run the SPMF library (something went wrong...)')


def _data_to_spmf_file_format_transaction_db(encoded_data, TEMP_DIR):
    """ Store the data in a transaction database to be used by spmf """
    f = NamedTemporaryFile(mode='w', dir=TEMP_DIR, delete=False, suffix='spmf_sdb.txt')
    for i, ed in enumerate(encoded_data):
        transaction = np.unique(ed).astype(str)
        f.write(' '.join(transaction))
        f.write('\n')
    f.close()
    return f.name


def _data_to_spmf_file_format_sequence_db(encoded_data, TEMP_DIR):
    """ Store the data in a sequence database to be used by spmf """
    f = NamedTemporaryFile(mode='w', dir=TEMP_DIR, delete=False, suffix='spmf_sdb.txt')
    # the following does not work for event logs
    #np.savetxt(f.name, encoded_data.astype(int), delimiter=' -1 ', fmt='%s', newline=' -2\n')
    for i, ed in enumerate(encoded_data):
        transaction = ed.astype(str)
        f.write(' -1 '.join(transaction))
        f.write(' -2\n')
    f.close()
    return f.name


def _spmf_output_to_patterns(output_file_name, pattern_type):
    """ Read the output of the SPMF file and return the found patterns

    SPMF itemset output:
    3 #SUP: 4
    1 3 #SUP: 3
    2 5 #SUP: 4
    2 3 5 #SUP: 3
    1 2 3 5 #SUP: 2

    SPMF sequential pattern output:
    4 -1 3 -1 2 -1 #SUP: 2
    5 -1 7 -1 3 -1 2 -1 #SUP: 2
    5 -1 1 -1 3 -1 2 -1 #SUP: 2
    where -1 is a separator
    """
    # read out the file
    with open(output_file_name, 'r') as file:
        spmf_lines = file.readlines()
    spmf_lines = [p.strip() for p in spmf_lines if p.strip() != '']

    # itemsets or sequential patterns
    if pattern_type == 'itemset':
        patterns, supports = _parse_itemset_patterns(spmf_lines)
    else:
        patterns, supports = _parse_sequential_patterns(spmf_lines)

    return patterns, supports


def _parse_itemset_patterns(spmf_lines):
    """ Parse the lines of the SPMF output to patterns in numpy array container
        --> encoded: still contains integer values
    """
    patterns = []
    supports = []
    for line in spmf_lines:
        new_s = int(line.split('#SUP:')[1])
        if new_s < 1:
            # impossible to have support < 1
            continue
        new_p = np.array(line.split('#SUP:')[0].strip(' ').split(' ')).astype(int)
        patterns.append(new_p)
        supports.append(new_s)
    return _sort_on_support(np.array(patterns), np.array(supports))

#Len: new 25/04/2019
def _sort_on_support(patterns,supports):
    ix_sort = np.argsort(supports)[::-1]
    patterns = patterns[ix_sort]
    supports = supports[ix_sort]
    return patterns,supports

def _parse_sequential_patterns(spmf_lines):
    """ Parse the lines of the SPMF output to patterns in numpy array container
        --> encoded: still contains integer values
    """
    patterns = []
    supports = []
    for line in spmf_lines:
        new_s = int(line.split('#SUP:')[1])
        if new_s < 1:
            # impossible to have support < 1
            continue
        new_p = np.array(list(filter(None, line.split('#SUP:')[0].strip(' ').split(' -1')))).astype(int)
        patterns.append(new_p)
        supports.append(new_s)
    # patterns are not sorted!
    return _sort_on_support(np.array(patterns), np.array(supports))


def data_encode_as_int(data):
    """ Encode the float/int data as ints
        --> vectorized with numpy (for efficiency)

        Note: this function works on: strings, floats, ints
        Note: this function does not reduce the current precision of the data
    """
    if isinstance(data, list):
        # event log: same encoding scheme, but different way to do it
        n = len(data)
        uniques = np.unique(np.concatenate(data))
        encode_dct = {u: i+1 for i, u in enumerate(uniques)}
        def _map_encode(v):
            return encode_dct[v]
        vectorize_func = np.vectorize(_map_encode)
        encoded_data = np.array(data.copy())
        for i, e in enumerate(data):
            if len(e) != 0:
                encoded_data[i] = vectorize_func(e)
    else:
        n, m = data.shape
        uniques = np.unique(data)
        encode_dct = {u: i+1 for i, u in enumerate(uniques)}
        def _map_encode(v):
            return encode_dct[v]
        vectorize_func = np.vectorize(_map_encode)
        encoded_data = vectorize_func(data.flatten()).reshape(n, m)
    return encode_dct, encoded_data


def patterns_decode_from_int(patterns, decode_dct):
    """ Decode the encoded patterns using the decoding dictionary
        --> vectorized as much as possible
    """
    def _map_decode(v):
        return decode_dct[v]
    vectorize_func = np.vectorize(_map_decode)
    decoded_patterns = []
    for p in patterns:
        decoded_patterns.append(vectorize_func(p))
    return np.array(decoded_patterns)


def make_temp_dir():
    """ Return a random directory in the ./temp/ folder """
    random_hex = uuid.uuid4().hex
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp', random_hex)
    if not(os.path.isdir(temp_dir)):
        os.makedirs(temp_dir)
    return temp_dir


def remove_temp_dir(temp_dir):
    """ Remove the directory (clean up) """
    try:
        shutil.rmtree(temp_dir)
    except:
        print('WARNING: could not remove the temporary path', temp_dir)
