export PYTHONPATH=/Users/lfereman/git2/pbad-public/src
echo $PYTHONPATH
python3 ../../src/main_TIPM.py -input ./lunges_and_sidelunges_vs_squats.csv -type all -columns PC1,PC2 -itemset_fnames ./lunges_and_sidelunges_vs_squats-PC1-itemsets.csv,./lunges_and_sidelunges_vs_squats-PC2-itemsets.csv -sequential_fnames ./lunges_and_sidelunges_vs_squats-PC1-sp.csv,./lunges_and_sidelunges_vs_squats-PC2-sp.csv

            
