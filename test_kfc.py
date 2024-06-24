"""
==============================================
Run KFC tests for finding the optimal k in a
collection of datasets
 
Jun 2024
==============================================
"""

#!/usr/bin/env python3

print(__doc__)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sys
import glob
import os
import re
import ntpath
import kfc

from sklearn.preprocessing import MinMaxScaler

inpath  = sys.argv[1]
perffile = sys.argv[2]
skip_header = int(sys.argv[3])

print("\nData folder:",inpath)
print("Performance file:",perffile)

results = {}

for idf, filename in enumerate(glob.glob(os.path.join(inpath, '*'))):
    print("\nData file", filename)
    print("Data file index: ", idf)

    d_name = ntpath.basename(filename)
    dataset = np.genfromtxt(filename, delimiter=',', skip_header=skip_header)
    X, ygt = dataset[:,0:-1], dataset[:,-1].astype(int)
    
    if -1 in np.unique(ygt):
        ygt[ygt>-1] = 0
        ygt[ygt==-1] = 1
    else: 
        ygt[ygt>0] = 1

    X = MinMaxScaler().fit_transform(X)

    n_samples = len(ygt)
    outliers_fraction = sum(ygt)/len(ygt)

    k_min,k_max=3,50
    k_candidates=range(k_min,k_max)
    optimal_k_by_kfcs,optimal_k_by_kfcr=kfc.KFC(k_candidates,X)
    print(d_name, optimal_k_by_kfcs,optimal_k_by_kfcr)
    results[d_name] = [optimal_k_by_kfcs,optimal_k_by_kfcr]

df = pd.DataFrame.from_dict(results)

df = df.T
df.columns = ['KFCS','KFCR']
print(df)

df.to_csv(perffile)
print('Peformances saved in:',perffile)

