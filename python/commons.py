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
import ast


classifiers = ['bayesnet', 'decisionstump', 'decisiontable', 'ibk', 'j48', 'jrip', 'kstar', 'lmt', 'logistic', 'multilayerperceptron', 'naivebayes', 'naivebayesmultinomial', 'oner', 'part', 'reptree', 'randomforest', 'randomtree', 'simplelogistic', 'smo', 'votedperceptron', 'zeror']
preprocessors = ['bestfirst_cfssubseteval', 'greedystepwise_cfssubseteval', 'ranker_correlationattributeeval', 'ranker_gainratioattributeeval', 'ranker_infogainattributeeval', 'ranker_onerattributeeval', 'ranker_principalcomponents',  'ranker_relieffattributeeval', 'ranker_symmetricaluncertattributeeval']


def map2dict(m):
    return ast.literal_eval(m.replace("=", "\": \"").replace(", ", "\", \"").replace("{", "{\"").replace("}", "\"}"))


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


def removeOutliersFromBatches(df, target="fittime", percentile=.75, maxDeviationFactor = 10):
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
                        q = np.quantile(rDF[target], percentile)
                        outliers = rDF.query("(" + target + " > 10 or " + str(q) + " > 10) and (" + target + " > " + str(q * maxDeviationFactor) + " or " + target + " < " + str(q / maxDeviationFactor) + ")")
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

def getRowWhereTrainPortionIsClosestTo(df, fitsize, epsilon = 100):
    bestMatch = None
    bestDistance = 100
    for i, row in df[["fitsize"]].iterrows():
        dist = np.abs(row["fitsize"] - fitsize)
        if dist < bestDistance:
            bestMatch = df.loc[i]
            bestDistance = dist
            if bestDistance < epsilon:
                break
    return bestMatch




def removeOutliersFromBatches(df, groupfeatures, targets, maxIQRFactor = 1.5, minRangeToDrop = 30):
    groups = df.groupby(groupfeatures)
    pbar = tqdm(total = len(groups))
    sizeAtStart = len(df)
    droppedIndices = []
    for gIndex, group in groups:
        bef = len(group) + 1
        cur = len(group)
        while len(group) > 0 and bef != cur:
            if len(group) < 4:
                dropFromGroup = group.index
            else:
                dropFromGroup = []
                for target in targets:
                    col = group[target]
                    q1 = np.quantile(col, 0.25)
                    q3 = np.quantile(col, 0.75) 
                    iqr = q3 - q1
                    if np.max(col) - np.min(col) > minRangeToDrop:
                        dropFromGroup.extend(group[group[target] > q3 + np.max([1, iqr]) * maxIQRFactor].index)
                        dropFromGroup.extend(group[group[target] < q1 - np.max([1, iqr]) * maxIQRFactor].index)
            droppedIndices.extend(set(dropFromGroup))
            group = group.drop(index = dropFromGroup)
            bef = cur
            cur = len(df)
        pbar.update(1)
    pbar.close()
    df = df.drop(index = set(droppedIndices))
    sizeAtEnd = len(df)
    print("Size after outlier removal: " + str(sizeAtEnd) + ". Removed: " + str(sizeAtStart - sizeAtEnd))
    return df


def plotDispersion(df, indexAttribute, batchAttributes, targets,  ax=None):
    
    if type(targets) != list:
        targets = [targets]
    
    # First compute dispersions
    indexValues = pd.unique(df[indexAttribute])
    stds = []
    bstds = []
    for index in tqdm(indexValues):
        group = df[df[indexAttribute] == index]
        stdMap = {}
        for target in targets:
            stdMap[target] = np.std(group[target])
        stds.append(stdMap)
        subGroups = group.groupby(batchAttributes)
        subStds = {}
        for target in targets:
            subStds[target] = []
        for j, subGroup in subGroups:
            for target in targets:
                subStds[target].append(np.std(subGroup[target]))
        bstdMap = {}
        for target in targets:
            bstdMap[target] = np.mean(subStds[target])
        bstds.append(bstdMap)
    
    # now plot dispersions
    if ax == None:
        fig, ax = plt.subplots(1, len(targets), figsize=(7 * len(targets),6))
    
    for i, target in enumerate(targets):
        ind = np.arange(len(stds))
        width=0.4
        a = np.ravel(ax)[i]
        a.barh(ind, [x[target] for x in stds], width, label="Overall std")
        a.barh(ind + width, [x[target] for x in bstds], width, color="red", label="Mean std within equivalence classes")
        a.set_yticks(range(len(indexValues)))
        a.set_yticklabels(indexValues, rotation=45)
        a.set_xscale("log")
        a.invert_yaxis()
        a.axvline(1, color="blue", linestyle="--", linewidth="1")
        a.axvline(10, color="green", linestyle="--", linewidth="1")
        a.legend()
    return ax






def readParameterFromOptionString(os, param, default):
    return os.split(" ")[os.split(" ").index("-" + param) + 1] if ("-" + param) in os else default

def readNumericParameterFromOptionString(os, param, default):
    return float(readParameterFromOptionString(os, param, default))

def readBinaryParameterFromOptionString(os, param):
    if pd.isnull(os):
        return 0
    return 1 if ("-" + param) in os.split(" ") else 0

def assureSearchEvalOptions(df):
    if not "searcheroptions" in df.columns:
        parts = [x.split(";") for x in df["algorithmoptions"].values]
        df["searcheroptions"] = [part[0] for part in parts]
        df["evaloptions"] = [part[1] for part in parts]

def explodeAlgorithmOptions(df, binarize=True):
    algos = pd.unique(df["algorithm"])
    if len(algos) > 1:
        raise Exception("There is information for more than one algorithm in the dataframe!")
    algorithm = algos[0]
    attributes = None
    df = df.copy()
        
    if algorithm == 'bayesnet':
        df["D"] = [ 1 if "-D" in s else 0 for s in df["algorithmoptions"].values]
        if binarize:
            df["Q_K2"] = [ 1 if "-Q weka.classifiers.bayes.net.search.local.K2" in s else 0 for s in df["algorithmoptions"].values]
            df["Q_Tabu"] = [ 1 if "-Q weka.classifiers.bayes.net.search.local.TabuSearch" in s else 0 for s in df["algorithmoptions"].values]
            df["Q_SA"] = [ 1 if "-Q weka.classifiers.bayes.net.search.local.SimulatedAnnealing" in s else 0 for s in df["algorithmoptions"].values]
            df["Q_LAGDHC"] = [ 1 if "-Q weka.classifiers.bayes.net.search.local.LAGDHillClimber" in s else 0 for s in df["algorithmoptions"].values]
            df["Q_TAN"] = [ 1 if "-Q weka.classifiers.bayes.net.search.local.TAN" in s else 0 for s in df["algorithmoptions"].values]
            df["Q_HC"] = [ 1 if "-Q weka.classifiers.bayes.net.search.local.HillClimber" in s else 0 for s in df["algorithmoptions"].values]
        else:
             df["Q"] = [readParameterFromOptionString(x, "Q", 'weka.classifiers.bayes.net.search.local.K2') for x in df["algorithmoptions"]]
    
    if algorithm == 'decisiontable':
        df["I"] = [ 1 if "-I" in s else 0 for s in df["algorithmoptions"].values]
        df["X"] = [float(x.split(" ")[-1]) if "-X" in x else 1 for x in df["algorithmoptions"]]
        if binarize:
            df["E_auc"] = [ 1 if "-E auc" in s else 0 for s in df["algorithmoptions"].values]
            df["E_mae"] = [ 1 if "-E mae" in s else 0 for s in df["algorithmoptions"].values]
            df["E_rmse"] = [ 1 if "-E rmse" in s else 0 for s in df["algorithmoptions"].values]
            df["E_acc"] = 1 - np.maximum(df["E_auc"], np.maximum(df["E_mae"], df["E_rmse"]))
            df["S_GSW"] = [ 1 if "-S weka.attributeSelection.GreedyStepwise" in s else 0 for s in df["algorithmoptions"].values]
        else:
            df["E"] = [readParameterFromOptionString(x, "E", 'acc') for x in df["algorithmoptions"]]
            df["S"] = [readParameterFromOptionString(x, "S", 'weka.attributeSelection.BestFirst') for x in df["algorithmoptions"]]
            
    
    if algorithm == 'ibk':
        for att in ["I", "F", "E", "X"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["algorithmoptions"]]
        df["K"] = [readNumericParameterFromOptionString(x, "K", 0) for x in df["algorithmoptions"]]
        return df
    
    if algorithm == "j48":
        #print(df)
        attributes = ["O", "U", "B", "S", "A", "C", "M"]
        for att in ["O", "U", "B", "S", "A"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["algorithmoptions"]]
        df["C"] = [readNumericParameterFromOptionString(x, "C", 0.25) for x in df["algorithmoptions"]]
        df["M"] = [readNumericParameterFromOptionString(x, "M", 2) for x in df["algorithmoptions"]]
    
    if algorithm == "jrip":
        attributes = ['E', 'P', "F", "N", "O"]
        for att in ["E", "P"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["algorithmoptions"]]
        df["F"] = [readNumericParameterFromOptionString(x, "F", 3) for x in df["algorithmoptions"]]
        df["N"] = [readNumericParameterFromOptionString(x, "N", 2) for x in df["algorithmoptions"]]
        df["O"] = [readNumericParameterFromOptionString(x, "O", 2) for x in df["algorithmoptions"]]

    if algorithm == 'lmt':
        attributes = ["B", "R", "C", "P", "A", "M", "W"]
        for a in ["B", "R", "C", "P", "A"]:
            df[a] = [ 1 if ("-" + a) in s else 0 for s in df["algorithmoptions"].values]
        df["M"] = [readNumericParameterFromOptionString(x, "M", 15) for x in df["algorithmoptions"]]
        df["W"] = [readNumericParameterFromOptionString(x, "W", 0) for x in df["algorithmoptions"]]
   
    if algorithm == "logistic":
        attributes = ["R"]
        df["R"] = [readNumericParameterFromOptionString(x, "R", 1e-8) for x in df["algorithmoptions"]]
    
    
    if algorithm == 'multilayerperceptron':
        for a in ["B", "R", "C", "D"]:
            df[a] = [ 1 if ("-" + a) in s else 0 for s in df["algorithmoptions"].values]
        df["L"] = [readNumericParameterFromOptionString(x, "L", 0.3) for x in df["algorithmoptions"]]
        df["M"] = [readNumericParameterFromOptionString(x, "M", 0.2) for x in df["algorithmoptions"]]
        if binarize:
            df["H_i"] = [ 1 if "-H i" in s else 0 for s in df["algorithmoptions"].values]
            df["H_o"] = [ 1 if "-H o" in s else 0 for s in df["algorithmoptions"].values]
            df["H_t"] = [ 1 if "-H t" in s else 0 for s in df["algorithmoptions"].values]
            df["H_a"] = 1 - np.maximum(df["H_i"], np.maximum(df["H_o"], df["H_t"]))
        else:
            df["H"] = [readParameterFromOptionString(x, "H", 'a') for x in df["algorithmoptions"]]
            
    if algorithm == "naivebayes":
        attributes = ['K', 'D']
        for att in attributes:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["algorithmoptions"]]
    
    if algorithm == "oner":
        attributes = ['B']
        df["B"] = [readNumericParameterFromOptionString(x, "B", 6) for x in df["algorithmoptions"]]
    
    if algorithm == "part":
        for att in ["R", "B", "U", "J"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["algorithmoptions"]]
        df["M"] = [readNumericParameterFromOptionString(x, "M", 2) for x in df["algorithmoptions"]]
        df["N"] = [readNumericParameterFromOptionString(x, "N", 3) for x in df["algorithmoptions"]]
    
    if algorithm =="randomforest" or algorithm == "randomtree":
        if algorithm == "randomtree":
            for att in ["U"]:
                df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["algorithmoptions"]]
        for att in ["B"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["algorithmoptions"]]
        
        for att in ["B"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["algorithmoptions"]]
        if algorithm == "randomforest":
            df["I"] = [readNumericParameterFromOptionString(x, "I", 100) for x in df["algorithmoptions"]]
        df["K"] = [readNumericParameterFromOptionString(x, "K", 0) for x in df["algorithmoptions"]]
        df["M"] = [readNumericParameterFromOptionString(x, "M", 1) for x in df["algorithmoptions"]]
        df["V"] = [readNumericParameterFromOptionString(x, "V", 0.001) for x in df["algorithmoptions"]]
        df["N"] = [readNumericParameterFromOptionString(x, "N", 0) for x in df["algorithmoptions"]]
    
    if algorithm == "reptree":
        for att in ["P"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["algorithmoptions"]]
        df["M"] = [readNumericParameterFromOptionString(x, "M", 2) for x in df["algorithmoptions"]]
        df["V"] = [readNumericParameterFromOptionString(x, "V", 1e-3) for x in df["algorithmoptions"]]
        df["L"] = [readNumericParameterFromOptionString(x, "L", -1) for x in df["algorithmoptions"]]
        df["N"] = [readNumericParameterFromOptionString(x, "N", 3) for x in df["algorithmoptions"]]
    
    if algorithm == "simplelogistic":
        for att in ["S", "A", "P"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["algorithmoptions"]]
        df["W"] = [readNumericParameterFromOptionString(x, "W", 0) for x in df["algorithmoptions"]]
        df["H"] = [readNumericParameterFromOptionString(x, "H", 50) for x in df["algorithmoptions"]]
        df["I"] = [readNumericParameterFromOptionString(x, "I", 0) for x in df["algorithmoptions"]]
        df["M"] = [readNumericParameterFromOptionString(x, "M", 500) for x in df["algorithmoptions"]]
        
    if algorithm == "smo":
        df["C"] = [readNumericParameterFromOptionString(x, "C", 1) for x in df["algorithmoptions"]]
        df["N"] = [readNumericParameterFromOptionString(x, "N", 0) for x in df["algorithmoptions"]]
        df["L"] = [readNumericParameterFromOptionString(x, "L", 0.001) for x in df["algorithmoptions"]]
        df["P"] = [readNumericParameterFromOptionString(x, "P", 1.0e-12) for x in df["algorithmoptions"]]
        df["V"] = [readNumericParameterFromOptionString(x, "V", -1) for x in df["algorithmoptions"]]
    
    if algorithm == "votedperceptron":
        df["I"] = [readNumericParameterFromOptionString(x, "I", 1) for x in df["algorithmoptions"]]
        df["E"] = [readNumericParameterFromOptionString(x, "E", 1) for x in df["algorithmoptions"]]
        df["M"] = [readNumericParameterFromOptionString(x, "M", 10000) for x in df["algorithmoptions"]]
    
    if algorithm == "bestfirst_cfssubseteval":
        assureSearchEvalOptions(df)
        df["D"] = [readNumericParameterFromOptionString(x, "D", 1) for x in df["searcheroptions"]]
        df["N"] = [readNumericParameterFromOptionString(x, "N", 1) for x in df["searcheroptions"]]
        df["S"] = [readNumericParameterFromOptionString(x, "S", 1) for x in df["searcheroptions"]]
        for att in ["L", "M", "Z"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["evaloptions"]]
    
    if algorithm == "greedystepwise_cfssubseteval":
        assureSearchEvalOptions(df)
        df["C"] = [readNumericParameterFromOptionString(x, "C", 1) for x in df["searcheroptions"]]
        df["B"] = [readNumericParameterFromOptionString(x, "B", 1) for x in df["searcheroptions"]]
        df["N"] = [readNumericParameterFromOptionString(x, "N", 1) for x in df["searcheroptions"]]
        for att in ["L", "M", "Z"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["evaloptions"]]
    
    if algorithm == "ranker_correlationattributeeval":
        assureSearchEvalOptions(df)
        df["N"] = [readNumericParameterFromOptionString(x, "N", 1) for x in df["searcheroptions"]]
    
    if algorithm == "ranker_gainratioattributeeval":
        assureSearchEvalOptions(df)
        df["N"] = [readNumericParameterFromOptionString(x, "N", 1) for x in df["searcheroptions"]]

    if algorithm == "ranker_infogainattributeeval":
        assureSearchEvalOptions(df)
        df["N"] = [readNumericParameterFromOptionString(x, "N", 1) for x in df["searcheroptions"]]
        for att in ["M", "B"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["evaloptions"]]
    
    if algorithm == "ranker_onerattributeeval":
        assureSearchEvalOptions(df)
        df["N"] = [readNumericParameterFromOptionString(x, "N", 1) for x in df["searcheroptions"]]
        for att in ["D"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["evaloptions"]]
        df["F"] = [readNumericParameterFromOptionString(x, "F", 1) for x in df["evaloptions"]]
        df["B"] = [readNumericParameterFromOptionString(x, "B", 1) for x in df["evaloptions"]]
        
    if algorithm == "ranker_principalcomponents":
        assureSearchEvalOptions(df)
        df["N"] = [readNumericParameterFromOptionString(x, "N", 1) for x in df["searcheroptions"]]
        df["A"] = [readNumericParameterFromOptionString(x, "A", 1) for x in df["evaloptions"]]
        df["R"] = [readNumericParameterFromOptionString(x, "R", 1) for x in df["evaloptions"]]
        for att in ["C", "O"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["evaloptions"]]
    
    if algorithm == "ranker_relieffattributeeval":
        assureSearchEvalOptions(df)
        df["N"] = [readNumericParameterFromOptionString(x, "N", 1) for x in df["searcheroptions"]]
        df["K"] = [readNumericParameterFromOptionString(x, "K", 1) for x in df["evaloptions"]]
        df["M"] = [readNumericParameterFromOptionString(x, "M", 1) for x in df["evaloptions"]]
        df["A"] = [readNumericParameterFromOptionString(x, "A", 2) for x in df["evaloptions"]]
        for att in ["W"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["evaloptions"]]

    if algorithm == "ranker_symmetricaluncertattributeeval":
        assureSearchEvalOptions(df)
        df["N"] = [readNumericParameterFromOptionString(x, "N", 1) for x in df["searcheroptions"]]
        for att in ["M"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["evaloptions"]]
    return df