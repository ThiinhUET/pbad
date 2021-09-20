import os
import numpy as np
from pattern_mining_spmf import mine_closed_itemsets, mine_closed_sequential_patterns, mine_closed_sequential_patterns_CM_CLASP
from pattern_mining_spmf import mine_maximal_itemsets, mine_minimal_infrequent_itemsets

def test_mine_closed_itemsets():
    #dataset init
    data = np.array([[1,2,3],
                     [1,2,3],
                     [1,2,3],
                     [3,4,5]])
    print(data)
    tempdir = '../../temp/unittest/'
    os.makedirs(tempdir, exist_ok=True)
    
    #mine support=1.0
    patterns, supports = mine_closed_itemsets(data, 1.0, tempdir)
    assert patterns == [[3]], "Should be [3]"
    
    #mine support=0.0
    patterns, supports = mine_closed_itemsets(data, 0.0, tempdir)
    print(patterns)
    print(supports)
    assert patterns[0].tolist() == [3], "Should be [3]"
    assert patterns[1].tolist() == [1,2,3], "Should be [1,2,3]"
    assert patterns[2].tolist() == [3,4,5], "Should be [3,4,5]"
    
def test_mine_closed_sequential_patterns():
    #dataset init
    data = np.array([[1,2,3,1],
                     [1,2,3,1],
                     [1,2,3,1],
                     [3,4,5,6]])
    print(data)
    tempdir = '../../temp/unittest/'
    os.makedirs(tempdir, exist_ok=True)
    
    #mine support=1.0
    patterns, supports = mine_closed_sequential_patterns(data, 1.0, tempdir)
    print(patterns)
    assert patterns == [[3]], "Should be [3]"
    
    #mine support=0.5
    patterns, supports = mine_closed_sequential_patterns(data, 0.5, tempdir)
    print(patterns)
    print(supports)
    assert patterns[0].tolist() == [3], "Should be [3]"
    assert patterns[1].tolist() == [1,2,3,1], "Should be [1,2,3,1]"

def test_mine_closed_sequential_patterns_BUG_CM_CLAPS():
    #dataset init
    data = np.array([[1,2,3,1],
                     [1,2,3,1],
                     [1,2,3,1],
                     [3,4,5,6]])
    print(data)
    tempdir = '../../temp/unittest/'
    os.makedirs(tempdir, exist_ok=True)
    
    #mine support=1.0
    patterns, supports = mine_closed_sequential_patterns_CM_CLASP(data, 1.0, tempdir)
    print(patterns)
    assert patterns == [[3]], "Should be [3]"
    
    #mine support=0.5
    patterns, supports = mine_closed_sequential_patterns_CM_CLASP(data, 0.5, tempdir)
    print(patterns)
    print(supports)
    assert patterns[0].tolist() == [3], "Should be [3]"
    assert patterns[1].tolist() == [1,2,3,1], "Should be [1,2,3,1] -> NOT [1,2,3]"
    
def test_mine_closed_sequential_patterns_BUG_Transaction_size_constant():
    #dataset init
    data = np.array([[1,2,3,1],
                     [1,2,3,1],
                     [1,2,3,1],
                     [3,4,5]]) #Transactions should not ALWAYS BE SAME SIZE
    print(data)
    tempdir = '../../temp/unittest/'
    os.makedirs(tempdir, exist_ok=True)
    
    #mine support=1.0
    patterns, supports = mine_closed_sequential_patterns(data, 1.0, tempdir)
    print(patterns)
    assert patterns == [[3]], "Should be [3]"
    
    #mine support=0.5
    patterns, supports = mine_closed_sequential_patterns(data, 0.5, tempdir)
    print(patterns)
    print(supports)
    assert patterns[0].tolist() == [3], "Should be [3]"
    assert patterns[1].tolist() == [1,2,3], "Should be [1,2,3]"

def test_mine_maximal_itemsets():
    #dataset init
    data = np.array([[1,2,3],
                     [1,2,3],
                     [1,2,3],
                     [3,4,5]])
    print(data)
    tempdir = '../../temp/unittest/'
    os.makedirs(tempdir, exist_ok=True)
    
    #mine support=1.0
    patterns, supports = mine_maximal_itemsets(data, 1.0, tempdir)
    print(patterns)
    assert patterns == [[3]], "Should be [3]"
    
    #mine support=0.1
    patterns, supports = mine_maximal_itemsets(data, 0.1, tempdir)
    print(patterns)
    print(supports)
    assert patterns[0].tolist() == [1,2,3], "Should be [1,2,3]"
    assert patterns[1].tolist() == [3,4,5], "Should be [3,4,5]"

def test_mine_maximal_itemsets_bug_support_0():
    #dataset init
    data = np.array([[1,2,3],
                     [1,2,3],
                     [1,2,3],
                     [3,4,5]])
    print(data)
    tempdir = '../../temp/unittest/'
    os.makedirs(tempdir, exist_ok=True)
    
    #mine support=1.0
    patterns, supports = mine_maximal_itemsets(data, 1.0, tempdir)
    print(patterns)
    assert patterns == [[3]], "Should be [3]"
    
    #mine support=0.1
    patterns, supports = mine_maximal_itemsets(data, 0.0, tempdir)
    print(patterns)
    print(supports)
    assert patterns[0].tolist() == [1,2,3], "Should be [1,2,3]"
    assert patterns[1].tolist() == [3,4,5], "Should be [3,4,5]"
  
def test_mine_minimal_infrequent_itemsets():
    #dataset init
    data = np.array([[1,2,3],
                     [1,2,3],
                     [1,5,9],
                     [3,4,5]])
    print(data)
    tempdir = '../../temp/unittest/'
    os.makedirs(tempdir, exist_ok=True)
    
    #mine support=1.0
    patterns, supports = mine_minimal_infrequent_itemsets(data, 1.0, tempdir)
    print(patterns)
    #assert patterns == [[3]], "Should be [3]"
    
    #mine support=0.1
    patterns, supports = mine_minimal_infrequent_itemsets(data, 0.3, tempdir)
    print(patterns)
    print(supports)
    #assert patterns[0].tolist() == [1,2,3], "Should be [1,2,3]"
    #assert patterns[1].tolist() == [3,4,5], "Should be [3,4,5]"
      
if __name__ == "__main__":
    test_mine_closed_itemsets()
    test_mine_closed_sequential_patterns()
    #test_mine_closed_sequential_patterns_BUG_CM_CLAPS()
    #test_mine_closed_sequential_patterns_BUG_Transaction_size_constant()
    test_mine_maximal_itemsets()
    #test_mine_maximal_itemsets_bug_support_0()
    test_mine_minimal_infrequent_itemsets()