"""
==============================================
Generate comparison plots from 

May 2024
==============================================
"""

#!/usr/bin/env python3

print(__doc__)

import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy.spatial.distance as distance
import mpl_toolkits.mplot3d.axes3d as axes3d
import random
import re

random.seed(10)

def xaxis_format(dfa, satype):

    if 'sa_size' in satype:
        scname = 'cardinality'
        colname = 'size (x1000)'
        xvals = (1+2*dfa['idata']*dfa['idata'])*1000 
        xticks = np.unique(xvals)
        xlabels = (xticks/1000).astype(int).astype(str)
        xlabels[0] = "" 
    elif satype == 'sa_dim':
        scname = 'dimensionality'
        colname = 'dimensions'
        xvals = 2+dfa['idata']*dfa['idata']*dfa['idata']  
        xticks = np.unique(xvals)
        xlabels = xticks.astype(str)
        xlabels[1] = "" 
        xlabels[2] = "" 
    elif satype == 'sa_outr':
        scname = 'outlier proportion'
        colname = 'outlier ratio (%)'
        xvals = 1+dfa['idata']*3 
        xticks = np.unique(xvals)
        xlabels = xticks.astype(str)
    elif satype == 'sa_ddif':
        scname = 'inliers-outliers density'
        colname = 'dens inls / dens outs'
        Rin = 0.1 + dfa['idata']*0.01
        Rout = 0.3 - dfa['idata']*0.01
        outdens = 3 / (np.pi * np.multiply(Rout-Rin, Rout-Rin))  
        inldens = 97 / (np.pi * np.multiply(Rin,Rin))
        xvals = np.divide(inldens,outdens).astype(int)
        xticks = np.unique(xvals)
        xlabels = xticks.astype(str)
        xlabels[1] = "" 
        xlabels[2] = "" 
    elif satype == 'sa_mdens':
        scname = 'density layers'
        colname = 'density layers'
        xvals = 2+dfa['idata']
        xticks = np.unique(xvals)
        xlabels = xticks.astype(str)
    elif satype == 'sa_clusts':
        scname = 'zonification'
        colname = 'clusters'
        xvals = 1+dfa['idata']
        xticks = np.unique(xvals)
        xlabels = xticks.astype(str)
    elif satype == 'sa_loc':
        scname = 'local outliers'
        colname = 'loc. outlier ratio (%)'
        xvals = 1+dfa['idata']*3 
        xticks = np.unique(xvals)
        xlabels = xticks.astype(str)
    
    return xvals, xticks, xlabels, colname, scname



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
df['idata'] = 0  
df['kpam'] = 0  
for index, row in df.iterrows():
    num = re.findall(r'\d+', row["dataset"])
    df.loc[index,'idata'] = num
    num = re.findall(r'\d+', row["alg"])
    df.loc[index,'kpam'] = num
    df.loc[index,'alg'] = ''.join([i for i in row['alg'] if not i.isdigit()])
    df.loc[index,'dataset'] = ''.join([i for i in row['dataset'] if not i.isdigit()])

df['idata'] = df['idata'].astype(int)
print(df)

algs = ["ABOD", "LOOP", "K-NN", "LOF","SDO"]
metrics = ["adj_Patn", "adj_maxf1", "adj_ap", "roc_auc", "AMI", "runtime"]
satypes = ['sa_size','sa_dim','sa_outr','sa_loc','sa_mdens','sa_ddif','sa_clusts']
ksens = [3, 5, 10, 20, 50]
y = 'adj_Patn'

for alg in algs:
    aux = df[df['alg']==alg]

    fig, axes = plt.subplots(1, 7, figsize=(22,4))
    #fig.suptitle(alg, fontsize=16)
    #satypes = np.unique(aux['dataset'].to_numpy())
    
    for i,satype in enumerate(satypes):
        auxsp = aux[aux['dataset']==satype].copy()

        sns.set_theme(style="white")
        xvals, xticks, xlabels, colname, scname = xaxis_format(auxsp, satype)
        auxsp[colname] = xvals

        if i<6:
            sns.lineplot(ax=axes[i], data=auxsp, x=colname, y=y, hue="kpam", palette="crest", linewidth=2, legend=False, markers=True, marker='o')
        else:
            sns.lineplot(ax=axes[i], data=auxsp, x=colname, y=y, hue="kpam", palette="crest", linewidth=2, legend=True, markers=True, marker='o')
            axes[i].legend(frameon=False, title="kpam")
            sns.move_legend(axes[i], "upper left", bbox_to_anchor=(1, 1))
            plt.setp(axes[i].get_legend().get_texts(), fontsize=14) 
            plt.setp(axes[i].get_legend().get_title(), fontsize=14) 

        axes[i].set_title(scname, fontsize=16) 
        axes[i].set_xlabel(colname, fontsize=16) 
        axes[i].set_ylabel(y, fontsize=16) 
        axes[i].set_xticks(xticks) 
        axes[i].set_xticklabels(xlabels)
        axes[i].set_ylim([0,1])
        axes[i].spines[['right', 'top']].set_visible(False)

        if i>0:
            axes[i].set_ylabel("")

    plt.tight_layout()
    plotname = resfolder + "/" + alg+ "_" + y + ".png"
    plt.savefig(plotname)
    plt.close()



aux = df[df['dataset']=='sa_size']
aux = aux[aux['alg']!='SDOa']
xvals, xticks, xlabels, colname, scname = xaxis_format(aux, 'sa_size')
aux[colname] = xvals
aux.rename(columns={"runtime": "runtime (s)"}, inplace=True)

sns.set_theme(style="white")
g = sns.relplot(
    data=aux,
    x=colname, y='runtime (s)', col="alg", hue="kpam",
    kind="line", palette="crest", linewidth=2, markers=True,
    col_wrap=5, height=3, aspect=1, legend=True, marker='o', facet_kws={'sharey': False, 'sharex': True}
)

#g.fig.suptitle("runtime", fontsize=16)
plt.xticks(xticks, labels=xlabels) 
g.tight_layout()
plotname = resfolder + "/" + "size_time.png"
plt.savefig(plotname)
plt.close()


aux = df[df['dataset']=='sa_dim']
aux = aux[aux['alg']!='SDOa']
xvals, xticks, xlabels, colname, scname = xaxis_format(aux, 'sa_dim')
aux[colname] = xvals
aux.rename(columns={"runtime": "runtime (s)"}, inplace=True)

sns.set_theme(style="white")
g = sns.relplot(
    data=aux,
    x=colname, y='runtime (s)', col="alg", hue="kpam",
    kind="line", palette="crest", linewidth=2, markers=True,
    col_wrap=5, height=3, aspect=1, legend=True, marker='o', facet_kws={'sharey': False, 'sharex': True}
)

#g.fig.suptitle("runtime", fontsize=16)
plt.xticks(xticks, labels=xlabels) 
g.tight_layout()
plotname = resfolder + "/" + "dim_time.png"
plt.savefig(plotname)
plt.close()

algorithm = "K-NN"
metric = "roc_auc"
df_kopt = df[df['alg']==algorithm][['dataset','idata','kpam',metric]]

kopt_knn = []
for d in df_kopt['dataset'].unique():
    for i in df_kopt['idata'].unique():
        aux = df_kopt[ df_kopt['dataset']==d ]
        aux = aux[ aux['idata']==i ]
        accs = aux[metric].to_numpy()
        ks = aux['kpam'].to_numpy()
        idxs = np.argwhere(accs == np.amax(accs)).flatten()
        kos = ks[idxs]
        kopt_knn.append([d,i,str(kos[0])+'-'+str(kos[-1])])

dfk = pd.DataFrame(kopt_knn, columns=['dataset','id','kpam']).sort_values(by=['dataset', 'id'])
filename = resfolder + "/" + "kNN_opt_ks.csv"
dfk.to_csv(filename, index=False)
