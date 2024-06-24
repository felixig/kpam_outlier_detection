"""
==============================================
Generate latex tables from KFCS and KFCR results 

May 2024
==============================================
"""

#!/usr/bin/env python3

print(__doc__)

import numpy as np
import pandas as pd
import sys
import os
import re


def get_group(i, satype):

    if 'sa_size' in satype:
        scname = 'cardinality'
        colname = 'size (x1000)'
        var_val = (1+2*i*i) 
    elif satype == 'sa_dim':
        scname = 'dimensionality'
        colname = 'dimensions'
        var_val = 2+i*i*i  
    elif satype == 'sa_outr':
        scname = 'outlier proportion'
        colname = 'outlier ratio (%)'
        var_val = 1+i*3 
    elif satype == 'sa_ddif':
        scname = 'inliers-outliers density'
        colname = 'dens inls / dens outs'
        Rin = 0.1 + i*0.01
        Rout = 0.3 - i*0.01
        outdens = 3 / (np.pi * np.multiply(Rout-Rin, Rout-Rin))  
        inldens = 97 / (np.pi * np.multiply(Rin,Rin))
        var_val = np.divide(inldens,outdens).astype(int)
    elif satype == 'sa_mdens':
        scname = 'density layers'
        colname = 'density layers'
        var_val = 2+i
    elif satype == 'sa_clusts':
        scname = 'zonification'
        colname = 'clusters'
        var_val = 1+i
    elif satype == 'sa_loc':
        scname = 'local outliers'
        colname = 'loc. outlier ratio (%)'
        var_val = 1+i*3 
    
    return scname, colname, var_val



## input arguments
inputfile = sys.argv[1]
resfolder = sys.argv[2]
currentpath = os.path.dirname(os.path.abspath(__file__))

## creating folders if missing
path = os.path.join(currentpath, resfolder) 
if os.path.exists(path):
    pass
else: 
    os.mkdir(path)    
print("Outputs saved in:", path)

df = pd.read_csv(inputfile)
df['i'] = 0  
df['group'] = 0  
df['variable'] = 0  
df['value'] = 0  
for index, row in df.iterrows():
    num = int(re.findall(r'\d+', row["dataset"])[0])
    df.loc[index,'i'] = num
    df.loc[index,'dataset'] = ''.join([i for i in row['dataset'] if not i.isdigit()])
    df.loc[index,'group'], df.loc[index,'variable'], df.loc[index,'value'] = get_group(num, df.loc[index,'dataset'])

df['i'] = df['i'].astype(int)
df2 = df.pivot(index='group',columns=['i'],values=['KFCS','KFCR'])
df3 = df.pivot(index='variable',columns=['i'],values=['value','value'])

df2.to_latex(resfolder+'/kfc_vals.tex')
df3.to_latex(resfolder+'/kfc_units.tex')

