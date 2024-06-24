"""
==============================================
Comparison of anomaly detection algorithms 
with multi-dimensional data with GT
 
Dec 2022
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

from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from scipy.special import erf

from pyod.models.abod import ABOD
#from pyod.models.hbos import HBOS
#from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
#from pyod.models.loci import LOCI
#from pyod.models.ocsvm import OCSVM
#from pyod.models.cof import COF
#from pyod.models.sod import SOD
from pysdo import SDO as SDOa
from sdoclust import SDO
from PyNomaly import loop
#from hdbscan import HDBSCAN #GLOSH

from indices import get_indices
import time

np.random.seed(100)

def abod(c,k):
    model = ABOD(contamination=c, n_neighbors=k, method='fast')
    return model
 
def knn(c,k):
    model = KNN(contamination=c, n_neighbors=k)
    return model

def lof(c,k):
    model = LOF(contamination=c, n_neighbors=k)
    return model

def sdo(c,k):
    model = SDO(x=k)
    return model

def sdoa(c,k):
    model = SDOa(contamination=c, x=k, return_scores=True)
    return model

def LoOP(c, k):
    model = loop
    return model


def select_algorithm(argument,c,k):
    switcher = {"ABOD":abod, "K-NN":knn, "LOF":lof, "SDO":sdo, "LOOP":LoOP, "SDOa":sdoa}
    model = switcher.get(argument, lambda: "Invalid algorithm")
    return model(c,k)

def normalize(s, method):
    if method=='abodreg':
        s = -1 * np.log10(s/np.max(s))
    if (method=='gauss' or method=='abodreg'):
        mu = np.nanmean(s)
        sigma = np.nanstd(s)
        s = (s - mu) / (sigma * np.sqrt(2))
        s = erf(s)
        s = s.clip(0, 1).ravel()
    elif method=='minmax':
        s = (s - s.min()) / (s.max() - s.min())
    return s

inpath  = sys.argv[1]
scfolder = sys.argv[2]
perffile = sys.argv[3]
norm = sys.argv[4]
skip_header = int(sys.argv[5])

#algs = ["ABOD", "HBOS", "iForest", "K-NN", "LOF", "OCSVM","SDO","LOOP","GLOSH"]
#cols = ["dataset","ABOD", "HBOS", "iForest", "K-NN", "LOF", "OCSVM","SDO","LOOP","GLOSH"]
algs = ["ABOD", "LOOP", "K-NN", "LOF","SDO","SDOa"]
#algs = ["K-NN", "SDO"]
cols = ["dataset","K-NN", "LOF", "LOOP", "ABOD", "SDO","SDOa"]
#cols = ["dataset", "K-NN", "SDO"]
metrics = ["adj_Patn", "adj_maxf1", "adj_ap", "roc_auc", "AMI", "runtime"]
ksens = [3, 5, 10, 20, 50]
reps = 5

currentpath = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(currentpath, scfolder) 
if os.path.exists(path):
    pass
else: 
    os.mkdir(path)    

print("\nData folder:",inpath)
print("Scores folder:",scfolder)
print("Performance file:",perffile)

dfpf = {}

for idf, filename in enumerate(glob.glob(os.path.join(inpath, '*'))):
    print("\nData file", filename)
    print("Data file index: ", idf)

    dfsc = {}

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

    ### OUTLIER DET. ALGORITHMS 
    for a_name in algs:
        for kparam in ksens:
            print("-----------------------------")
            print("Algorithm, k-param:", a_name, kparam)

            for rep in range(reps):

                start_time = time.time()

                algorithm = select_algorithm(a_name,outliers_fraction,kparam)
                if a_name == "LOOP":
                    scores = algorithm.LocalOutlierProbability(X, extent=2, n_neighbors=kparam, use_numba=True).fit().local_outlier_probabilities.astype(float)
                    scores = normalize(scores, norm)
                    threshold = np.quantile(scores, 1-outliers_fraction)
                    y = (scores > threshold)*1
                else:        
                    algorithm.fit(X)
                    if a_name == "SDO" or a_name == "SDOa":
                        scores = algorithm.predict(X)
                        scores = normalize(scores, norm)
                        threshold = np.quantile(scores, 1-outliers_fraction)
                        y = (scores > threshold)*1
                    else:
                        y = algorithm.predict(X)
                        scores = algorithm.decision_scores_
                        if (a_name == "ABOD" and norm == "gauss"):
                            scores = normalize(scores, "abodreg")
                        else:
                            scores = normalize(scores, norm)

                end_time = time.time()

                ami, apatn, amf1, aap, roc, ami, trun = [],[],[],[],[],[],[]

                trun = end_time - start_time
                ami.append(adjusted_mutual_info_score(ygt, y))
                res = get_indices(ygt, scores)
                apatn.append(res['adj_Patn'])
                amf1.append(res['adj_maxf1'])
                aap.append(res['adj_ap'])
                roc.append(res['auc'])
                print(rep)

            AMI = np.nanmean(ami)
            runtime = np.nanmean(trun)
            APTN = np.nanmean(apatn)
            AMF1 = np.nanmean(amf1)  
            AAP = np.nanmean(aap)  
            ROC = np.nanmean(roc) 
           
            print("Adj P@n: %.2f, Adj MaxF1: %.2f, Adj AP: %.2f, ROC-AUC: %.2f, AMI: %.2f, runtime: %.2f" % (APTN, AMF1, AAP, ROC, AMI, runtime) )

            dfpf[d_name+a_name+str(kparam)] = [APTN, AMF1, AAP, ROC, AMI, runtime]
            dfsc[a_name+str(kparam)] = scores

    df = pd.DataFrame.from_dict(dfsc)
    df.to_csv(scfolder+'/'+d_name, index=False)
    print('Scores saved in:',(scfolder+'/'+d_name))

df = pd.DataFrame.from_dict(dfpf)
#columns = df.columns
df = df.T
df.columns = metrics
#df.index = columns

df.to_csv(perffile)
print('Peformances saved in:',perffile)

