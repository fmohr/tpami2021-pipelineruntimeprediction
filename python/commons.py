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
from tqdm.notebook import tqdm
import ast
from logging import *
import pickle
from libruntimepredictionanalysis import *

FILE_CLASSIFIERS_DEFAULT = "data/rawruntimes/classifierresults-default.csv"
FILE_CLASSIFIERS_PARAMETRIZED = "data/rawruntimes/classifierresults-parametrized.csv"
FILE_PREPROCESSORS_DEFAULT = "data/rawruntimes/preprocessorresults-default.csv"
FILE_PREPROCESSORS_PARAMETRIZED = "data/rawruntimes/preprocessorresults-parametrized.csv"
FILE_METAFEATURES = "data/metafeatures.csv"
FILE_DATASETS = "data/datasets.csv"

datasets = [    3,     6,    12,    14,    16,    18,    21,    22,    23, 24,    26,    28,    30,    31,    32,    36,    38,    44, 46,    57,    60,   179,   180,   181,   182,   183,   184, 185,   273,   293,   300,   351,   354,   357,   389,   390, 391,   392,   393,   395,   396,   398,   399,   401,   554, 679,   715,   718,   720,   722,   723,   727,   728,   734, 735,   737,   740,   741,   743,   751,   752,   761,   772, 797,   799,   803,   806,   807,   813,   816,   819,   821, 822,   823,   833,   837,   843,   845,   846,   847,   849, 866,   871,   881,   897,   901,   903,   904,   910,   912, 913,   914,   917,   923,   930,   934,   953,   958,   959, 962,   966,   971,   976,   977,   978,   979,   980,   991, 993,   995,  1000,  1002,  1018,  1019,  1020,  1021,  1036, 1037,  1039,  1040,  1041,  1042,  1049,  1050,  1053,  1059, 1067,  1068,  1069,  1111,  1112,  1114,  1116,  1119,  1120, 1128,  1130,  1134,  1138,  1139,  1142,  1146,  1161,  1166, 1216,  1242,  1457,  1485,  1486,  1501,  1569,  4136,  4137, 4541,  4552, 23380, 23512, 40497, 40685, 40691, 40900, 40926, 40927, 40971, 40975, 41026, 41064, 41065, 41066, 41143, 41146, 41164, 41946, 41991]

coredatasets = [3, 57, 741, 743, 772, 813, 903, 904, 914, 923]

classifiers = ['bayesnet', 'decisionstump', 'decisiontable', 'ibk', 'j48', 'jrip', 'kstar', 'lmt', 'logistic', 'multilayerperceptron', 'naivebayes', 'naivebayesmultinomial', 'oner', 'part', 'reptree', 'randomforest', 'randomtree', 'simplelogistic', 'smo', 'votedperceptron', 'zeror']

preprocessors = ['bestfirst_cfssubseteval', 'greedystepwise_cfssubseteval', 'ranker_correlationattributeeval', 'ranker_gainratioattributeeval', 'ranker_infogainattributeeval', 'ranker_onerattributeeval', 'ranker_principalcomponents',  'ranker_relieffattributeeval', 'ranker_symmetricaluncertattributeeval']

algorithms = classifiers + preprocessors
parametrizablealgorithms = [a for a in algorithms if not a in ["decisionstump", "kstar", "naivebayesmultinomial", "zeror"]]

metalearners = ["adaboostm1", "bagging", "logitboost", "randomcommittee", "randomsubspace"]

regressors = ["linear", "ann", "forest"]

algorithmshortcuts = {
    'bestfirst_cfssubseteval': 'se-bf',
    'greedystepwise_cfssubseteval': "se-gs",
    'ranker_correlationattributeeval': "cr",
    'ranker_gainratioattributeeval': "gr",
    'ranker_infogainattributeeval': "ig",
    'ranker_onerattributeeval': "or",
    'ranker_principalcomponents': "pca",
    'ranker_relieffattributeeval': 're',
    'ranker_symmetricaluncertattributeeval': 'su',
    'bayesnet': 'bn',
    'decisionstump': "ds",
    'decisiontable': "dt",
    'ibk': "ibk",
    'j48': "j48",
    'jrip': "jrip",
    'kstar': "k*",
    'lmt': "lmt",
    'logistic': "lr",
    'multilayerperceptron': "ann",
    'naivebayes': "nb",
    'naivebayesmultinomial': "nbm",
    'oner': "1-r",
    'part': "part",
    'randomforest': "rf",
    'randomtree': "rt",
    'reptree': "rep",
    'simplelogistic': "sl",
    'sl': "sl2",
    'smo': "smo",
    'votedperceptron': "vp",
    'zeror': "0-r"
}

featureAliases = {
    'fitsize': "ni",
    'numattributes': "na",
    'numlabels': "nl",
    'numnumericattributes': "nn",
    'numsymbolicattributes': "ns",
    'numberofcategories': "nc",
    'numericattributesafterbinarization': "nab",
    'totalvariance': "tv",
    'attributestocover50pctvariance': "v50",
    'attributestocover90pctvariance': "v90",
    'attributestocover95pctvariance': "v95",
    'attributestocover99pctvariance': "v99"
}
learnerAliases = {
    'ann': 'ANN',
    'forest': 'RF',
    'linear': 'LR'
}



# instance-attribute grid
def getEvaluationGridPoints():
    datapoints = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 14000, 15000, 16000, 18000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 250000, 500000, 750000, 1000000]
    attributes = [5, 10, 25, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000, 18000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    P = []
    lim = 3 * 10**8
    for x in datapoints:
        for y in attributes:
            prod = (x + 1500) * y
            if prod <= lim:
                P.append((x,y))
    return np.array(list(set(P)))

def getRegressor(learner):
    return getRegressionAlgorithm(learner)

def getRegressionAlgorithm(learner):
    if learner == 'forest':
        reg = RandomForestRegressor(n_estimators=100)
    else:
        if learner == "ann":
            reg = sk.neural_network.MLPRegressor(max_iter=1000)
        else:
                if learner == "linear":
                    reg = sk.linear_model.LinearRegression()
                else:
                    if learner == "isotonic":
                        reg = IsotonicRegression()
                    else:
                        raise Exception("Unknown learner " + learner)
    return reg

def map2dict(m):
    return ast.literal_eval(m.replace("=", "\": \"").replace(", ", "\", \"").replace("{", "{\"").replace("}", "\"}"))




def getRandomAlgorithmOptions(algorithm, n, seed=None):
    
    B = [True, False]
    isClassifier = algorithm in classifiers
    
    if algorithm == 'bayesnet':
        options = {
            "D": B,
            "Q": ["weka.classifiers.bayes.net.search.local." + q for q in ["K2", "TabuSearch", "SimulatedAnnealing", "LAGDHillClimber", "TAN", "HillClimber"]]
        }
    elif algorithm == 'decisiontable':
        options = {
            "I": B,
            "E": ["acc", "rmse", "mae", "auc"],
            "S": ["weka.attributeSelection.BestFirst", "weka.attributeSelection.GreedyStepwise"],
            "X": list(range(1, 11))
        }
    elif algorithm == 'ibk':
        options = {
            "K": [2, 4, 8, 16, 32, 64],
            "I": B,
            "E": B,
            "X": B
        }
    elif algorithm == "j48":
        options = {
            "A": B,
            "S": B,
            "B": B,
            "U": B,
            "O": B,
            "C": [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            "M": [1, 4, 8, 16, 32, 64]
        }
    elif algorithm == "jrip":
        options = {
            "P": B,
            "E": B,
            "F": [1, 2, 3, 4, 5],
            "N": [1, 2, 3, 4, 5],
            "O": [1, 2, 4, 8, 16, 32, 64]
        }
    elif algorithm == 'lmt':
        options = {
            "B": B,
            "R": B,
            "C": B,
            "P": B,
            "A": B,
            "M": [1, 2, 4, 8, 16, 32, 64],
            "W": [0, 0.5, 1, 1.5, 2, 4]
        }
    elif algorithm == "logistic":
         options = {
            "R": [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0, 1.0, 10, 100]
        }
    elif algorithm == 'multilayerperceptron':
        options = {
            "B": B,
            "R": B,
            "C": B,
            "D": B,
            "L": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "M": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "H": ["a", "i", "o", "t"]
        }
    elif algorithm == "naivebayes":
        options = {
            "D": B,
            "K": B
        }
    elif algorithm == "oner":
        options = {
            "B": [1, 2, 4, 6, 8, 16, 32, 64]
        }
    elif algorithm == "part":
        options = {
            "R": B,
            "B": B,
            "U": B,
            "J": B,
            "M": [1, 2, 4, 8, 16, 32, 64],
            "N": [None, 2, 4, 8, 10, 16] # default is 3, which is enabled by None
        }
    elif algorithm =="randomforest" or algorithm == "randomtree":
        options = {
            "B": B,
            "K": list(range(11)),
            "M": [2, 4, 8, 16, 32, 64, 128],
            "V": ["0.000001", "0.00001", "0.0001", "0.001", "0.01", "0.1", "1", "10", "100"],
            "N": [0, 2, 4, 8, 10, 16]
        }
        if algorithm == "randomforest":
            options["I"] = [1, 2, 4, 8, 16, 32, 64, 128]
    elif algorithm == "reptree":
        options = {
            "P": B,
            "M": [2, 4, 8, 16, 32, 64, 128],
            "V": ["0.000001", "0.00001", "0.0001", "0.001", "0.01", "0.1", "1", "10", "100"],
            "L": [1, 2, 4, 8, 16, 32, 64],
            "N": [0, 2, 4, 8, 16]
        }
    elif algorithm == "simplelogistic":
        options = {
            "S": B,
            "A": B,
            "P": B,
            "W": [0, 0.5, 1, 1.5, 2],
            "H": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            "I": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            "M": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        }
        
    elif algorithm == "smo":
        options = {
            "C": ["0.000001", "0.00001", "0.0001", "0.001", "0.01", "0.1", "1", "10", "100", "1000", "10000"],
            "N": [1, 2],
            "L": ["0.000001", "0.00001", "0.0001", "0.001", "0.01", "0.1", "1", "10", "100"],
            "P": ["1.0e-14", "1.0e-13", "1.0e-11", "1.0e-10", "1.0e-9", "1.0e-8", "1.0e-7", "1.0e-6", "1.0e-5", "1.0e-4", "1.0e-3"],
            "V": [-1] + list(range(1, 11))
        }
    
    elif algorithm == "votedperceptron":
        options = {
            "I": ["1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"],
            "E": ["1", "2", "3", "4", "5"],
            "M": ["1", "10", "100", "1000", "10000", "100000", "1000000"]
        }
    
    elif algorithm == "bestfirst_cfssubseteval":
        options = {
            "search": {
                "D": [0, 1, 2],
                "N": [1, 10, 50, 100, 1000],
                "S": [0, 1, 2, 3]
            },
            "eval": {
                "L": B,
                "M": B,
                "Z": B
            }
        }
    elif algorithm == "greedystepwise_cfssubseteval":
        options = {
            "search": {
                "C": B,
                "B": B,
                "N": list(range(1, 11))
            },
            "eval": {
                "L": B,
                "M": B,
                "Z": B
            }
        }
    elif "ranker" in algorithm:
        options = {
            "search": {
                "N": list(range(1, 11)) + [20, 50, 100]
            }
        }
        if algorithm == "ranker_correlationattributeeval":
            options["eval"] = {}

        elif algorithm == "ranker_gainratioattributeeval":
            options["eval"] = {}

        elif algorithm == "ranker_infogainattributeeval":
            options["eval"] = {
                "M": B,
                "B": B
            }

        elif algorithm == "ranker_onerattributeeval":
            options["eval"] = {
                "F": [2, 5, 10],
                "D": B,
                "B": list(range(1, 21))
            }

        elif algorithm == "ranker_principalcomponents":
            options["eval"] = {
                "R": [0.5, 0.6, 0.75, 0.8, 0.9, 0.95, 0.99],
                "C": B,
                "O": B
            }

        elif algorithm == "ranker_relieffattributeeval":
            options["eval"] = {
                "K": ["1", "2", "4", "10", "100"],
                "A": ["1", "2", "3", "10"],
                "M": ["1", "2", "10", "100", "1000"],
                "W": B
            }

        elif algorithm == "ranker_symmetricaluncertattributeeval":
            options["eval"] = {
                "M": B
            }
        else:
            raise Exception("Unsupported pre-processing algorithm for parameter generation: " + str(algorithm))
    
    else:
        raise Exception("Unsupported algorithm for parameter generation: " + str(algorithm))
    
    # now conduct latin hypercube sampling in space
    if isClassifier:
        newSamples = n
        out = []
        it = 0
        while newSamples > 0:
            candidates = getLHCRandomOptionStrings(options, newSamples, None if (seed is None and it == 0) else (seed + it if (not seed is None) else it))
            newSamples = 0
            for cand in candidates:
                if isConfigurationValid(algorithm, cand):
                    out.append(cand)
                else:
                    newSamples +=1
            it += 1
        return out
                
    else:
        searchOptions = getLHCRandomOptionStrings(options["search"], n, seed)
        evalOptions = getLHCRandomOptionStrings(options["eval"], n, seed)
        return [searchOptions[i] + ";" + evalOptions[i] for i in range(n)]
    
def getLHCRandomOptionStrings(options, n, seed):
    r = random.Random(seed)
    samples = np.empty((n, len(options)),dtype=object)
    for i, option in enumerate(options):
        domain = options[option]
        
        # if the domain is smaller than the number of samples, draw each candidate equally often
        items = []
        if len(domain) <= n:
            while len(items) < n:
                items.extend(domain)
            r.shuffle(items)
            items = items[:n]
        else:
            stepsizes = [int(np.floor(len(domain) / n)) for k in range(n)]
            modoffset = len(domain) % n
            covered = []
            offset = 0
            for j in range(n):
                stepsize = stepsizes[j]
                if j < modoffset:
                    stepsize += 1
                subdomain = domain[offset : offset + stepsize]
                offset += stepsize
                covered.extend(subdomain)
                items.append(r.choice(subdomain))
        if len(items) != n:
            raise Exception()
        samples[:,i] = np.array(items)
        
    # compute option strings
    keys = list(options.keys())
    out = []
    for sample in samples:
        outstringarray = []
        for k, v in enumerate(sample):
            if not v is None:
                if str(v) in ["True", "False"]:
                    if v == True:
                        outstringarray.append("-" + keys[k])
                else:
                    outstringarray.append("-" + keys[k] + " " + str(v))
        ostring = " ".join(outstringarray)
        out.append(ostring)
    
    # return list of option strings
    if len(out) != n:
        raise Exception("Output has too few elements!")
    return out

def isConfigurationValid(algorithm, configstring):
    items = configstring.split(" ")
    
    if algorithm == "j48":
        if "-U" in items:
            
            if "-S" in items: # Subtree raising doesn't need to be unset for unpruned tree!
                return False
            if "-C" in items: # Doesn't make sense to change confidence for unpruned tree
                return False
        
    
    if algorithm == 'naivebayes':
        if "-D" in items and "-K" in items: # Can't use both kernel density estimation and discretization
            return False
        return True
    
    if algorithm == 'part':
        isReducedPruning = "-R" in items
        isFoldsDefined = "-N" in items
        if not isReducedPruning and isFoldsDefined:
            return False
    return True




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

def getAlgorithmOptions(algo):
    dfAlgo = dfConfigured[dfConfigured["algorithm"] == algo]
    if len(dfAlgo) == 0:
        return []
    standardCols = dfConfigured.columns
    colsAfterExpansion = explodeAlgorithmOptions(dfAlgo.iloc[:5].copy())
    return [x for x in colsAfterExpansion.columns if not x in standardCols and x != "searcheroptions" and x != "evaloptions"]


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
    if type(df) != pd.DataFrame:
        raise Exception("Invalid type. Expected dataframe but saw " + str(type(df)))
    if not "searcheroptions" in df.columns:
        parts = [x.split(";") for x in df["algorithmoptions"].values]
        df["searcheroptions"] = [part[0] for part in parts]
        df["evaloptions"] = [part[1] if len(part) > 1 else "" for part in parts]
        
def explodeAlgorithmOptions(df, algorithm="auto", binarize=True):
    if algorithm == 'auto':
        algos = pd.unique(df["algorithm"])
        if len(algos) == 0:
            raise Exception("No algorithms found in dataframe with " + str(len(df)) + " entries.")
        if len(algos) > 1:
            raise Exception("There is information for more than one algorithm in the dataframe!")
        algorithm = algos[0]
    attributes = None
    df = df.copy()
        
    if algorithm == 'bayesnet':
        df["D"] = [ 1 if "-D" in s else 0 for s in df["algorithmoptions"].values]
        if binarize:
            df["Q_K2"] = [ 1 if ("-Q weka.classifiers.bayes.net.search.local.K2" in s or not "-Q" in s) else 0 for s in df["algorithmoptions"].values]
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
        df["C"] = [readBinaryParameterFromOptionString(x, "C") for x in df["searcheroptions"]]
        df["B"] = [readBinaryParameterFromOptionString(x, "B") for x in df["searcheroptions"]]
        df["N"] = [readNumericParameterFromOptionString(x, "N", 10) for x in df["searcheroptions"]]
        for att in ["L", "M", "Z"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["evaloptions"]]
    
    if algorithm == "ranker_correlationattributeeval":
        assureSearchEvalOptions(df)
        df["N"] = [readNumericParameterFromOptionString(x, "N", 10) for x in df["searcheroptions"]]
    
    if algorithm == "ranker_gainratioattributeeval":
        assureSearchEvalOptions(df)
        df["N"] = [readNumericParameterFromOptionString(x, "N", 10) for x in df["searcheroptions"]]

    if algorithm == "ranker_infogainattributeeval":
        assureSearchEvalOptions(df)
        df["N"] = [readNumericParameterFromOptionString(x, "N", 10) for x in df["searcheroptions"]]
        for att in ["M", "B"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["evaloptions"]]
    
    if algorithm == "ranker_onerattributeeval":
        assureSearchEvalOptions(df)
        df["N"] = [readNumericParameterFromOptionString(x, "N", 10) for x in df["searcheroptions"]]
        for att in ["D"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["evaloptions"]]
        df["F"] = [readNumericParameterFromOptionString(x, "F", 1) for x in df["evaloptions"]]
        df["B"] = [readNumericParameterFromOptionString(x, "B", 1) for x in df["evaloptions"]]
        
    if algorithm == "ranker_principalcomponents":
        assureSearchEvalOptions(df)
        df["N"] = [readNumericParameterFromOptionString(x, "N", 10) for x in df["searcheroptions"]]
        df["A"] = [readNumericParameterFromOptionString(x, "A", 1) for x in df["evaloptions"]]
        df["R"] = [readNumericParameterFromOptionString(x, "R", 1) for x in df["evaloptions"]]
        for att in ["C", "O"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["evaloptions"]]
    
    if algorithm == "ranker_relieffattributeeval":
        assureSearchEvalOptions(df)
        df["N"] = [readNumericParameterFromOptionString(x, "N", 10) for x in df["searcheroptions"]]
        df["K"] = [readNumericParameterFromOptionString(x, "K", 1) for x in df["evaloptions"]]
        df["M"] = [readNumericParameterFromOptionString(x, "M", 1) for x in df["evaloptions"]]
        df["A"] = [readNumericParameterFromOptionString(x, "A", 2) for x in df["evaloptions"]]
        for att in ["W"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["evaloptions"]]

    if algorithm == "ranker_symmetricaluncertattributeeval":
        assureSearchEvalOptions(df)
        df["N"] = [readNumericParameterFromOptionString(x, "N", 10) for x in df["searcheroptions"]]
        for att in ["M"]:
            df[att] = [readBinaryParameterFromOptionString(s, att) for s in df["evaloptions"]]
            
    if algorithm in ["adaboostm1", "bagging", "logitboost", "randomcommittee", "randomsubspace"]:
        df["I"] = [readNumericParameterFromOptionString(x, "I", 10) for x in df["algorithmoptions"]]
        if algorithm in ["adaboostm1", "bagging", "logitboost", "randomsubspace"]:
            df["P"] = [readNumericParameterFromOptionString(x, "P", 1) for x in df["algorithmoptions"]]
        if algorithm == "adaboostm1":
            df["Q"] = [readBinaryParameterFromOptionString(s, "Q") for s in df["algorithmoptions"]]
        if algorithm == "logitboost":
            df["L"] = [readNumericParameterFromOptionString(x, "L", 1) for x in df["algorithmoptions"]]
            df["H"] = [readNumericParameterFromOptionString(x, "H", 1) for x in df["algorithmoptions"]]
            df["Z"] = [readNumericParameterFromOptionString(x, "Z", 1) for x in df["algorithmoptions"]]
    return df


def updateValidationPredictions(algorithm = None, parametrized=False, defaultobservations=False, globalModel = True, checkOnDuplicates = False):
    
    if defaultobservations:
        FILENAME = FILENAME_PREDICTIONS_POSTERIOR
        parametrized = True
    elif parametrized:
        FILENAME = FILENAME_PREDICTIONS_PARAMETRIZED
    else:
        FILENAME = FILENAME_PREDICTIONS_DEFAULT
    
    dfResults = pd.read_csv(FILENAME, delimiter=";") if path.exists(FILENAME) else None
    if not dfResults is None:
        dfResults.astype({"truth_fit": str, "predictions_fit": str, "truth_app": str, "predictions_app": str})
    hasChanged = False
    
    # compute grid and compatibility maps
    mainIndex = dfGrid["algorithmoptions"].notna() if parametrized else dfGrid["algorithmoptions"].isna()
    if not algorithm is None:
        mainIndex &= dfGrid["algorithm"] == algorithm
    
    for algorithm, dfAlgoOrig in dfGrid[mainIndex].groupby("algorithm"):
        
        dfResultsOnAlgorithm = None if dfResults is None else dfResults[dfResults["algorithm"] == algorithm]
        
        if not dfResultsOnAlgorithm is None:
            print(algorithm.upper(), len(dfResultsOnAlgorithm), "results are already in cache.")
        
        # append a column with default runtimes if necessary
        if defaultobservations:
            dfAlgo = getPosteriorJoin(dfAlgoOrig)
        else:
            dfAlgo = dfAlgoOrig
            
        # build connection to the status table
        dfStatus = STATUS_TABLE_COMPLETE[STATUS_TABLE_COMPLETE["algorithmoptions"].notna() if parametrized else STATUS_TABLE_COMPLETE["algorithmoptions"].isna()]
        dfAlgo = dfAlgo.merge(dfStatus.query("forglobalmodel == " + str(globalModel) + " and status == 'ok'"), on=["algorithm", "algorithmoptions", "openmlid", "fitsize", "fitattributes", "seed"])
        
        for dataset in tqdm(datasets):
            
            dsMatchIndex = dfAlgo["openmlid"] == dataset
            hasValueIndex = dfAlgo["fittime"].notna()
            
            #print("openmlid", dataset, "with algorithm", algorithm, "Observations on OTHER datasets: ", len(dfAlgo[dfAlgo["openmlid"] != dataset]), "Observations on this dataset:",len(dfAlgo[dfAlgo["openmlid"] == dataset]))
            
            # extract the train (all non d-instances of the original data) and validation (all d-instances for grid point of interest)
            trainsize = 100000
            oneMask = np.ones(len(dfAlgo)) if trainsize >= len(dfAlgo) else np.array(list(np.ones(trainsize)) + list(np.zeros(len(dfAlgo) - trainsize)))
            
            # get mask for all the validation point entries
            validationIndices = None
            availablecombos = []
            for combo in validationCombos: ## calidationCombos is a global constant
                pointmask = (dfAlgo["fitsize"] == combo[0]) & (dfAlgo["fitattributes"] == combo[1])
                if dfResultsOnAlgorithm is None or np.count_nonzero((dfResultsOnAlgorithm["fitsize"] == combo[0]) & (dfResultsOnAlgorithm["fitattributes"] == combo[1])) == 0:
                    validationIndices = pointmask if validationIndices is None else (validationIndices | pointmask)
                else:
                    #print("Ignoring indices for which results are already stored.")
                    pass
            if validationIndices is None:
                #print("No validation indices found. This is may be due to the fact that all validation results are already in the cache. Skipping.")
                continue
            
            # compute basic indices for training and validation
            trainIndices = ~dsMatchIndex & ~validationIndices & hasValueIndex & oneMask
            validIndices = dsMatchIndex & validationIndices & hasValueIndex
            
            print("Identified", np.count_nonzero(trainIndices), "rows for training and", np.count_nonzero(validIndices), " for validation.")
            
            # check on duplicates
            if checkOnDuplicates:
                overoffer = 0
                for gIndex, group in dfAlgo[trainIndices].groupby(["fitsize", "fitattributes"]):
                    if list(gIndex) in G:
                        overoffer += max(0, len(group) - 10)
                    else:
                        overoffer += len(group)
                print("Have " + str(overoffer) + " overoffered")
            
            # compute reduced dataset and also reduced results matrix
            dfAlgoDatasetInc = dfAlgo[dsMatchIndex]
            dfAlgoDatasetExc = dfAlgo[~dsMatchIndex]
            #dfResultsOnAlgorithmAndDataset = None if dfResultsOnAlgorithm is None else dfResultsOnAlgorithm[dfResultsOnAlgorithm["openmlid"] == dataset]
            #if not dfResultsOnAlgorithmAndDataset is None:
#                print("CACHE contains " + str(len(dfResultsOnAlgorithmAndDataset)))
            
            # if we have to make at least prediction we have not generated so far, do it now
            print("After checking the cache, there are", np.count_nonzero(validIndices), "remaining")
            
            # check whether the training set is complete (all points of grid contained)
            gridpointsCoveredByTrainData = [list(i) for i, g in dfAlgo[trainIndices].groupby(["fitsize", "fitattributes"])]
            missingGridPoints = [gp for gp in G if not gp in gridpointsCoveredByTrainData and not gp in validationCombos]
            unexpectedGridPoints = [gp for gp in gridpointsCoveredByTrainData if not gp in G]
            #print("Examples: ", (np.count_nonzero(trainIndices) - overoffer), "for training and", (np.count_nonzero(trainIndices)))
            #print("Missing: ", missingGridPoints)
            #print("Unexpected: ", unexpectedGridPoints)
            totalMissing = 10 * len(missingGridPoints)
            print("Missing grid points: " + str(len(missingGridPoints)))
            print("Covered grid points: " + str(len([gp for gp in gridpointsCoveredByTrainData if gp in G])))
            for gIndex, group in dfAlgo[trainIndices].groupby(["fitsize", "fitattributes"]):
                if list(gIndex) in G and not list(gIndex) in validationCombos:
                    missingForPoint = 10 - min(10, len(group))
                    totalMissing += missingForPoint
                    #print(gIndex, missingForPoint)
            print("Number of missing observations in total: ", totalMissing)

            # if there are validations missing, learn a model and obtain the validation
            if np.count_nonzero(validIndices) > 0:
                dfResultsTmp = trainAndGetPredictions(dfAlgo, ["forest"], trainIndices, validIndices, algorithmfeatures = parametrized, defaultobservations = defaultobservations, serializeModel=False)[["learner", "algorithm", "features", "openmlid", "fitsize", "fitattributes", "truth_fit", "predictions_fit", "truth_app", "predictions_app"]]
                for col in ["truth_fit", "predictions_fit", "truth_app", "predictions_app"]:
                    replacement = ['[' + ','.join([str(v) for v in a]) + ']' for a in dfResultsTmp[col].values]
                    dfResultsTmp[col] = replacement
                dfResults = dfResultsTmp if dfResults is None else pd.concat([dfResults, dfResultsTmp], axis=0)
                hasChanged = True
            else:
                print("NO VALIDATION INSTANCES AVAILABLE FOR THIS FOLD, SKIPPING.")
    
    if hasChanged:
        print("Saving changes ...")
        dfResults.to_csv(FILENAME, sep=";", index=False)
    return dfResults