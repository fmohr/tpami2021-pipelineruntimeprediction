import numpy as np
import numpy.ma as ma
import scipy as sp
import pandas as pd
import sklearn as sk
import itertools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ARDRegression
import random
import os.path
from os import path
from itertools import chain, combinations
from tqdm import tqdm_notebook as tqdm


def normalize(arr):
    min = np.min(arr, axis=0)
    max = np.max(arr, axis=0)
    range = max - min
    copy = np.subtract(arr, min)
    copy = np.divide(copy, range)
    return copy


def standardize(arr):
    mean = np.average(arr, axis=0)
    covariances = np.cov(arr.T, bias=True)
    copy = np.subtract(arr, mean)
    stdDev = np.sqrt(np.diag(covariances))
    copy = np.divide(copy, stdDev)
    return copy, mean, stdDev


## First, split meta knowledge about data into two
def prepareRuntimeKnowledge():
    df = pd.read_csv("classifier_runtimes.csv")
   #df = df[df["exception"].isnull()].drop("exception", axis=1)
    
    # Set traintime to full traintime +1 (3601) if the exception is a timeout or is dominated by another one that timed out
    timeouts = []
    for i, row in df[df["exception"].notnull()].query("traintime >= 0").iterrows():
        exception = str(row["exception"])
        if "Timeout" in exception and row["traintime"] >= 3600:
            df.at[i, "exception"] = np.nan
            timeouts.append((row["openmlid"], row["classifier"], row["trainpoints"]))

            #if "Memory" in exception:
        #    print("Memory exception detected")
    for i, row in df[df["exception"].notnull()].iterrows():
        exception = str(row["exception"])
        if "canceled due to" in exception:
            t = (row["openmlid"], row["classifier"], row["trainpoints"])
            isDominatedDueToTimeout = False
            for t2 in timeouts:
                # if the datapoint is dominated by another one due to a timeout, set the traintime to that time
                 if t2[0] == t[0] and t2[1] == t[1] and t2[2] < t[2]:
                    df.at[i, "traintime"] = 3601
                    df.at[i, "exception"] = np.nan
                    break
    df = df.query("traintime >= 0")
    df = df[df["exception"].isnull()]
    print(len(df))
    
    # now read in the meta features to prepare a join
    mf = pd.read_csv("metafeatures.csv", delimiter=";")
    m1Cols = ["openmlid", "datapoints_fold1", "seed"]
    m1Cols.extend(mf.filter(regex='^f1_',axis=1).columns)
    mfA = mf[m1Cols]
    renaming = {}
    for col in m1Cols:
        renaming[col] = col.replace('f1_', '')
    renaming["datapoints_fold1"] = "datapoints"
    mfA = mfA.rename(columns=renaming).dropna()
    
    m2Cols = ["openmlid", "datapoints_fold2", "seed"]
    m2Cols.extend(mf.filter(regex='^f2_',axis=1).columns)
    mfB = mf.query("datapoints_fold1 != datapoints_fold2")[m2Cols]
    renaming = {}
    for col in m2Cols:
        renaming[col] = col.replace('f2_', '')
    renaming["datapoints_fold2"] = "datapoints"
    mfB = mfB.rename(columns=renaming).dropna()
    mfDecomposed = pd.concat([mfA, mfB])
    if len(mfDecomposed) > 2 * len(mf):
        raise Exception("Decomposed meta features are more than twice the number of rows of meta features. Expected " + str(2 * len(mf)) + " but is " + str(len(mfDecomposed)) + " = "  + str(len(mfA)) + " + " + str(len(mfB)))
    idf = df.merge(mfDecomposed, left_on=["openmlid", "trainpoints", "seed"], right_on=["openmlid", "datapoints", "seed"]).drop_duplicates()
    #if len(idf) > len(df):
    #    raise Exception("Implausible result. Data must not increase! However, it increased from " + str(len(df)) + " to " + str(len(idf)))
    print("Combining " + str(len(df)) + " available classifier runtimes with " + str(len(mfA) + len(mfB)) + "/" + str(2 * len(mf)) + " available meta-features. This resulted in " + str(len(idf)) + " many informed rows.")
    idf.to_csv("classifierruntimes_with_meta_features.csv", index=False)


def removeOutliersFromBatches(df):
    print("Size before: " + str(len(df)))
    algorithms = pd.unique(df["algorithm"])
    datasets = pd.unique(df["openmlid"])
    pbar = tqdm(total = len(algorithms) * len(datasets))
    for c in algorithms:
        cDF = df.query("algorithm == '" + c + "'")
        numDatasets = datasets.shape[0]
        for d in datasets:
            dDF = cDF.query("openmlid == " + str(d))
            fitsizes = pd.unique(dDF["fitsize"])
            fitattributes = pd.unique(dDF["fitattributes"])
            for size in fitsizes:
                for atts in fitattributes:
                    rDF = dDF.query("fitsize == " + str(size) + " and fitattributes == " + str(atts))
                    if len(rDF) >= 4:
                        q3 = np.quantile(rDF["fittime"], .75)
                        outliers = rDF.query("fittime > " + str(q3 * 10))
                        if len(outliers) > 0:
                            df = df.drop(index = outliers.index)
            pbar.update(1)
    pbar.close()
    print("Size after outlier removal: " + str(len(df)))
    return df

#################
## Data access ##
#################
def filterDS(df, openmlid="", trainpoints="", seed=""):
    query = []
    if openmlid != "":
        query.append("openmlid == '" + str(openmlid) + "'")
    if trainpoints != "":
        query.append("trainpoints == " + str(trainpoints))
    if seed != "":
        query.append("seed == '" + str(seed) + "'")
    return df.query(' and '.join(query))

def getIndicesOfRowsForOpenMLId(df, openmlid):
    col = df[["openmlid"]].values[:,0]
    indices, cIndices = np.where(col != openmlid)[0], np.where(col == openmlid)[0]
    if len(indices) + len(cIndices) != len(df):
        raise Exception("The dataspace has not been partitioned.")
    return indices, cIndices


def implode(iterable, delimiter=","):
    return delimiter.join([str(x) for x in iterable])

def powerset(iterable):
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

def getRowWhereTrainPortionIsClosestTo(df, trainportion, epsilon = 0.01):
    bestMatch = None
    bestDistance = 100
    for i, row in df[["datapoints_x", "trainpoints"]].iterrows():
        portion = row["trainpoints"] / row["datapoints_x"]
        dist = np.abs(portion - trainportion)
        if dist < bestDistance:
            bestMatch = df.loc[i]
            bestDistance = dist
            if bestDistance < epsilon:
                break
    return bestMatch