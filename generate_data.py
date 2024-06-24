"""
==============================================
Generation of data for studying outlier scoring 
algorithms when facing different types of data
perturbation/variation
 
Jan 2023
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
from matplotlib.ticker import EngFormatter
import random

random.seed(10)

## input arguments
datafolder = sys.argv[1]
plotfolder = sys.argv[2]
currentpath = os.path.dirname(os.path.abspath(__file__))

## creating folders if missing
path = os.path.join(currentpath, datafolder) 
if os.path.exists(path):
    pass
else: 
    os.mkdir(path)    
print("Datasets saved in:", path)

path = os.path.join(currentpath, plotfolder) 
if os.path.exists(path):
    pass
else: 
    os.mkdir(path)    
print("Example plots saved in:", path)


def remove_closest(X,k,kcl):
    [m, n] = X.shape
    index = np.random.permutation(m)
    kpos = index[0:k] 
    O = X[kpos]

    dist = distance.cdist(O,X)
    dist_sorted = np.argsort(dist, axis=1)
    closest = dist_sorted[:,0:kcl].flatten()

    closest = np.unique(closest)
    for e in kpos:
        closest = np.delete(closest, np.argwhere(closest == e))

    y = np.zeros(m)
    y[kpos] = 1
    Xnew = np.delete(X,closest,0)
    ynew = np.delete(y,closest)

    return Xnew,ynew

def cluster_gen_levs(dim, size, inlD, outD, levs):

    X = np.empty([1,dim])
    y = np.empty(1)
    reduc = 1.9
    psizeacum = 0
    levD = outD/(levs-1)

    for i in range(levs):
        if i==0:
            psize = int(size/reduc)
        elif i==levs-1:
            #psize = int(size/(np.log10(10+i*i)*levs))    
            psize = size - psizeacum 
        else: 
            psize = psize - int(psize/reduc) 

        psizeacum += psize

        yp = i*np.ones(psize)

        if i==0:
            dXp = np.random.uniform(-inlD, inlD, psize)
        else:
            dXp = np.random.uniform(-levD, levD, psize)
            dXp[dXp>0] += inlD+levD*(i-1)
            dXp[dXp<0] -= inlD+levD*(i-1)

        Xp = np.random.normal(size=(psize,dim))
        Xp = Xp / np.linalg.norm(Xp,axis=1)[:, None]
        Xp = Xp * dXp[:, None]

        X = np.vstack((X, Xp))
        y = np.concatenate((y, yp), axis=0)

    y = np.nan_to_num(y)
    ind = np.random.permutation(len(y))
    X = X[ind,:]
    y = y[ind]

    columns = ["f"+str(i) for i in range(dim)]
    df = pd.DataFrame(X, columns = columns)
    df['y'] = y.astype(int)
    return df
  
def cluster_gen(dim, size, outr, inlD, outD):
    out_size = int(size*outr)
    inl_size = int(size*(1-outr))
    size = out_size + inl_size

    Xin, yin = np.zeros((inl_size,dim)), np.zeros(inl_size)
    Xout, yout = np.zeros((out_size,dim)), np.ones(out_size)

    dXin = np.random.uniform(-inlD, inlD, inl_size)
    dXout = np.random.uniform(-outD, outD, out_size) 
    dXout[dXout>0]+=inlD
    dXout[dXout<0]-=inlD

    Xin = np.random.normal(size=(inl_size,dim))
    Xout = np.random.normal(size=(out_size,dim))
    Xin = Xin / np.linalg.norm(Xin,axis=1)[:, None]
    Xin = Xin * dXin[:, None]
    Xout = Xout / np.linalg.norm(Xout,axis=1)[:, None]
    Xout = Xout * dXout[:, None]

    X = np.vstack((Xin, Xout))
    y = np.concatenate((yin, yout), axis=0)
    ind = np.random.permutation(size)
    X = X[ind,:]
    y = y[ind]

    columns = ["f"+str(i) for i in range(dim)]
    df = pd.DataFrame(X, columns = columns)
    df['y'] = y.astype(int)
    return df

def cluster_gen_holes(dim, size, outr, inlD, xclos):
    out_size = int(size*outr)
    inl_size_exp = int(size*(1-outr))
    inl_size = inl_size_exp + out_size*xclos

    Xin, yin = np.zeros((inl_size,dim)), np.zeros(inl_size)
    dXin = np.random.uniform(-inlD, inlD, inl_size)

    Xin = np.random.normal(size=(inl_size,dim))
    Xin = Xin / np.linalg.norm(Xin,axis=1)[:, None]
    Xin = Xin * dXin[:, None]

    X,y = remove_closest(Xin,out_size,xclos)

    ind = np.random.permutation(len(y))
    X = X[ind,:]
    y = y[ind]

    inliers = np.argwhere(y == 0)
    num_out = np.sum(y)
    num_inl = len(inliers)
    exceed = int(num_inl - (size - num_out))
    X = np.delete(X,inliers[:exceed],0)
    y = np.delete(y,inliers[:exceed])

    columns = ["f"+str(i) for i in range(dim)]
    df = pd.DataFrame(X, columns = columns)
    df['y'] = y.astype(int)
    return df

# Sens. analysis: dim
satypes = ['sa_dim', 'sa_size', 'sa_outr', 'sa_ddif', 'sa_mdens', 'sa_clusts', 'sa_loc']
reps = 8
np.random.seed(5) 

for satype in satypes:

    # base dataset config
    dim = 2
    size = 1000
    outr = 0.03
    inlD = 0.1
    outD = 0.3

    for i in range(reps):

        if satype=='sa_dim':
            dim = 2 + 2*i*i*i
        elif satype=='sa_size':
            size = 1000 + i*i*2000
        elif satype=='sa_outr':
            outr = 0.01 + i*0.03
            outD = 0.2 + i*0.03
        elif satype=='sa_ddif':
            inlD = 0.1 + i*0.01
            outD = 0.3 - i*0.01
        if satype=='sa_mdens' or satype=='sa_clusts' or satype=='sa_loc':
            pass
        else: 
            print("dim: %d, size: %d, outr: %f, inlD: %f, outD: %f" % (dim, size, outr, inlD, outD))
            df = cluster_gen(dim, size, outr, inlD, outD) 

        if satype=='sa_mdens':
            size, inlD, outD, levs = 10000, 0.1, 1, i+2
            df = cluster_gen_levs(dim, size, inlD, outD, levs)
            maxy = np.max(df['y'])
            print("dim: %d, size: %d, outr: %f, inlD: %f, outD: %f, levs: %d, last-layer: %d" % (dim, len(df), np.sum(df['y']>0)/len(df), inlD, outD, levs, np.sum(df['y']==maxy)))

        elif satype=='sa_clusts':
            clusters = i+1
            cols = ["f"+str(i) for i in range(dim)]
            df = pd.DataFrame(columns = cols+['y'])

            a = np.random.permutation(3*clusters)-int(1.5*clusters)
            b = np.random.permutation(3*clusters)-int(1.5*clusters)
            for c in range(clusters):
                dfp = cluster_gen(dim, size, outr, inlD, outD) 
                dfp['f0'] = dfp['f0'] * (1+c*0.5)
                dfp['f1'] = dfp['f1'] * (1+c*0.5)
                dfp['f0'] += a[c]
                dfp['f1'] += b[c]
                df = pd.concat([df,dfp], axis=0)
            df = df.sample(frac=1)
            print("dim: %d, size: %d, outr: %f, inlD: %f, outD: %f, clusters: %d" % (dim, len(df), outr, inlD, outD, clusters))

        elif satype=='sa_loc': 
            outr = 0.01 + i*0.03
            xclos = 15
            df = cluster_gen_holes(dim, size, outr, inlD, xclos) 
            print("dim: %d, size: %d, outr: %f, inlD: %f, xclos: %d" % (dim, len(df), sum(df.y)/len(df), inlD, xclos))

        name =  satype + str(i)
        df.to_csv(datafolder + '/' + name + '.csv', index=False)

nums = [0,0,1,7,6,7,5,7,1]
for i,satype in enumerate(['sa_size','sa_size']+satypes):
    name =  satype + str(nums[i])
    df = pd.read_csv(datafolder + '/' + name + '.csv')

    if satype=='sa_dim':

        fig = plt.figure(figsize=(5,4))
        ax = fig.add_subplot(111, projection = '3d')
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["tab:grey","tab:red"])

        x = df['f0']
        y = df['f1']
        z = df['f2']
        label = df['y']

        ax.set_xlabel("f0")
        ax.set_ylabel("f1")
        ax.set_zlabel("f2")

        ax.scatter(x, y, z, c=label, s=20, cmap=cmap, edgecolors='white')
        ax = plt.gca()
        ax.set_facecolor('white')
        leg = plt.legend()
        ax.get_legend().set_visible(False)

    elif satype=='sa_mdens':
        palette = ['blue','royalblue', 'violet', 'orange', 'gold', 'lightcoral', 'y']
        sns.scatterplot(data=df, x='f0', y='f1', hue='y', s=20, palette=palette)
        sns.set(rc={"figure.figsize":(6, 5)}) #width=3, #height=4
        sns.set_theme(style='white')

    else:
        g = sns.scatterplot(data=df, x='f0', y='f1', hue='y', s=20, legend=False, palette=['tab:grey','tab:red'])
        fmin = min([min(df['f0']),min(df['f1'])]) 
        fmax = max([max(df['f0']),max(df['f1'])]) 
        g.set_xlim([fmin, fmax])
        g.set_ylim([fmin, fmax])
        sns.set(rc={"figure.figsize":(5, 4)}) #width=3, #height=4
        sns.set_theme(style='white')

        if satype=='sa_loc':
            ax = plt.gca()
            formatter1 = EngFormatter(places=1, sep="\N{THIN SPACE}")  # U+2009
            ax.xaxis.set_major_formatter(formatter1)
            ax.yaxis.set_major_formatter(formatter1)

    plt.tight_layout()
    plt.savefig(plotfolder + '/' + name + '.png', dpi=200)
    plt.close()


