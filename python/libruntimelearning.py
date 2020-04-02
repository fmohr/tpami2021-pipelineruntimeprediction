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
from commons import *
from tqdm import tqdm_notebook as tqdm

########################
## Feature Expansions ##
########################
def expandMult(arr):
    numAttributes = arr.shape[1]
    numPairs = int(sp.special.binom(numAttributes, 2))
    expansion = np.empty((arr.shape[0], numPairs))
    c = 0
    for i in range(numAttributes):
        for j in range(i):
            v = arr[:,i] * arr[:,j]
            expansion[:, c] = v
            c += 1
    return np.concatenate((arr, expansion), axis=1)

def expandPower(arr, exp):
    return np.concatenate((arr, np.power(arr, exp)), axis=1)

def expandLog(arr):
    return np.concatenate((arr,  ma.log(arr)), axis=1)

############################
## Classifier Experiments ##
############################
def getLearnFile(algorithm, openmlids=[]):
    regressionFile = "learnfile-" + algorithm + ".csv"
    if not path.exists(regressionFile):
        L = ["forest", "ann", "linear"]
        E = powerset(["log", "pow", "mul"])
        TrSize = np.unique([int(1 + ((i+1)**2.5)) for i in range(50)])
        S = range(10) # seeds
        algorithms = [algorithm]
        C = np.array([x for x in itertools.product(openmlids, algorithms, TrSize, S, E)])
        z = np.zeros(C.shape[0]) - 1
        C = np.column_stack([range(C.shape[0]), C])
        cols = ["reg_id", "openmlid", "classifier", "traindatasize", "seed", "expansions"]
        for learner in L:
            cols.append("mse_" + learner + "_trainruntime_half")
            cols.append("mse_" + learner + "_trainruntime_full")
            cols.append("mse_" + learner + "_testruntime_half")
            cols.append("mse_" + learner + "_testruntime_full")
            C = np.column_stack([C, z, z, z, z])
        exp = pd.DataFrame(C, columns=cols)
        exp.to_csv(regressionFile, index=False)
        return exp
    else:
        return pd.read_csv(regressionFile)

def updateLearnFile(algorithm, results):
    regressionFile = "learnfile-" + algorithm + ".csv"
    df = getLearnFile(algorithm).T ## assume that file exists.
    ## IN THE df OBJECT, line indices correspond to reg_ids
    for i, r in results.iterrows():
        df[r["reg_id"]] = r.values
    df = df.T
    df.to_csv(regressionFile, index=False)
    
def getRegressionInputRepresentation(dataspace, basefeatures, openmlid, classifier, expansion, target):

    # reduce data only to those that have the desired classifier
    classifierSpace = dataspace.query("classifier == '" + classifier + "'")
    if len(classifierSpace.query("openmlid == " + str(openmlid))) == 0:
        raise Exception("The classifier space has no values for openmlid " + str(openmlid))
    classifierSpace = classifierSpace.dropna(subset=[target])
    
    # create features for regression problem
    X = np.nan_to_num(classifierSpace[basefeatures].values)
    if "log" in expansion:
        X = expandLog(X)
    if "pow" in expansion:
        X = expandPower(X, 2)
    if "mul" in expansion:
        X = expandMult(X)
    Y = classifierSpace[[target]].values
    Y = np.log(Y + 1)
    
    indices, cIndices = getIndicesOfRowsForOpenMLId(classifierSpace, openmlid)
    if len(cIndices) == 0:
        raise Exception("No test data selected for openmlid " + str(openmlid))
    
    # create train/test split for regression problem based on a leave-one-out-split (using the openmlid)
    Xtrain = X[indices]
    Ytrain = Y[indices]
    #print("Reducing train data to those that do NOT have openmlid " + str(openmlid) + ". Length: " + str(len(Xtrain)) + "/" + str(len(X)))
    #print(classifierSpace.iloc[indices])
    #print(Xtrain)
    #print(Ytrain)
    
    # reduce test set to those corresponding to half/full dataset (according to targetdssize)
    Xtest = X[cIndices]
    Ytest = Y[cIndices]
    return Xtrain, Ytrain, Xtest, Ytest

### The function runExperiment performs the following three steps:
###
### 1) trains a learner for the train or prediction time (based on the variable $target)
###     - based on all given data for the classifier that does NOT belong to the given openmlid,
###     - deriving features based on the expansion keyword (log, multiplication, polynomial features)
###     - using traindatasize many random samples from that set
### 2) predicts the runtimes on the GIVEN openmlid for the requested train sizes and FULL
###    (refering to the maximum number of training samples for which a ground truth is known)
###
### 3) returns a dictionary with the different sample sizes for which predictions are requested
###    each one consisting of a pair of ground truth values and prediction values
###
### Two targets are possible: "traintime" and "predictiontimeperinstance"
### In both cases, the traindatasize variable refers to the number of instances used for FITTING the algorithm
### The concrete number of instances for which predictions are made is not relevant and not considered.
def runExperiment(dataspace, basefeatures, openmlid, classifier, learner, traindatasize, expansion, target, samplesizes):
    Xtrain, Ytrain, Xtest, Ytest = getRegressionInputRepresentation(dataspace, basefeatures, openmlid, classifier, expansion, target)
    
    numInstancesCol = 0 ## num instances are in first column of X
    numInstances = Xtest[:,numInstancesCol]
    if numInstances.size == 0:
        return {}
    
    # now reduce the train set to the specified value
    tIndices = list(range(Xtrain.shape[0]))
    random.shuffle(tIndices)
    trainSizeBefore = len(Xtrain)
    Xtrain = Xtrain[tIndices[:traindatasize]]
    Ytrain = Ytrain[tIndices[:traindatasize]]
    #print("Reducing train data from " + str(trainSizeBefore) + " to " + str(traindatasize) + ". Effective train size: " + str(len(Xtrain)))
    
    # create learner
    if learner == 'forest':
        reg = RandomForestRegressor(n_estimators=100)
    else:
        if learner == "ann":
            reg = sk.neural_network.MLPRegressor(max_iter=1000)
        else:
                if learner == "linear":
                    reg = sk.linear_model.LinearRegression()
                else:
                    raise Exception("Unknown learner " + learner)
            
    
    # train learner
    #print ("Training " + str(reg) + " with " + "\n\t" + str(Xtrain) + "\nand\n\t" + str(Ytrain))
    reg.fit(Xtrain, np.ravel(Ytrain))
  
    # gather predictions for learner
    out = {}
    maxInstances = int(np.max(numInstances))
    for sampleSize in samplesizes:
        #print("GT: " + str(trueRuntimesFull))
        #print("PR: " + str(predRuntimesFull))
        #fullContainsInf = len(np.where(predRuntimesFull == pInf)[0]) > 0 
        #errorRateTrainFull = sk.metrics.mean_squared_error(trueRuntimesFull, predRuntimesFull) if not fullContainsInf else pInf
        if sampleSize == "full":
            sampleSize = maxInstances
        relevantIndices = np.where(numInstances == sampleSize)[0]
        if len(relevantIndices) == 0:
            out[sampleSize] = ([], [])
        else:
            XtestSampleSize = Xtest[relevantIndices, :].reshape(relevantIndices.size,Xtest.shape[1])
            YtestSampleSize = Ytest[relevantIndices, :].reshape(relevantIndices.size,1)
            out[sampleSize] = (np.ravel(np.exp(YtestSampleSize)), np.exp(reg.predict(XtestSampleSize)))
        if sampleSize == maxInstances:
            out["full"] = out[sampleSize]
    return out

# This is to recover runtime predictions and ground truths from the archive.
def getGTPredictionPairSeries(learner, basefeatures, expansions, trainpoints):
    serializedDF = pd.read_csv("runtime_predictions.csv").query("learner == '" + str(learner) + "' and basefeatures == '" + str(basefeatures).replace("'", "\\'").replace(",", ";") + "' and expansions == '" + str(expansions) + "' and trainpoints_algorithm == 'full' and trainpoints_learner == " + str(trainpoints))
    datasets = list(pd.unique(serializedDF["openmlid"]))
    classifiers = list(pd.unique(serializedDF["algorithm"]))
    seeds = list(pd.unique(serializedDF["seed"]))
    trainGTPredictionPairs = np.empty((len(datasets), len(classifiers)), dtype=object)
    testGTPredictionPairs = np.empty((len(datasets), len(classifiers)), dtype=object)
    pbar = tqdm(total=len(datasets) * len(classifiers) * len(seeds))
    for i, ds in enumerate(datasets):
        for j, c in enumerate(classifiers):
            if trainGTPredictionPairs[i][j] != None:
                raise Exception("ALREADY HAVE VALUES")
            trainVals = []
            testVals = []
            for seed in seeds:
                pbar.update(1)
                relevantData = serializedDF.query("openmlid == " + str(ds) + " and algorithm == '" + str(c) + "' and seed == " + str(seed))
                if len(relevantData) > 0:
                    if len(relevantData) >= 4:
                        trainGT = np.array(relevantData.query("recordtype == 'traintime_gt'")["entries"].values[0].split(";")).astype('float64')
                        trainPR = np.array(relevantData.query("recordtype == 'traintime_pr'")["entries"].values[0].split(";")).astype('float64')
                        testGT = np.array(relevantData.query("recordtype == 'predictiontime_gt'")["entries"].values[0].split(";")).astype('float64')
                        testPR = np.array(relevantData.query("recordtype == 'predictiontime_pr'")["entries"].values[0].split(";")).astype('float64')
                        if len(trainGT) != len(trainPR):
                            raise Exception("GT and PR of traintimes have different length!")
                        if len(testGT) != len(testPR):
                            raise Exception("GT and PR of prediction times have different length!")
                        trainVals.append((trainGT, trainPR))
                        testVals.append((testGT, testPR))
            trainGTPredictionPairs[i][j] = trainVals
            testGTPredictionPairs[i][j] = testVals
    pbar.close()
    return datasets, classifiers, trainGTPredictionPairs, testGTPredictionPairs

def updateGTPredictionPairSeries(datasets, classifiers, mfdf, learner, seeds, trainpoints, basefeatures, expansions, sampleSizes):
    serializedDF = pd.read_csv("runtime_predictions.csv")
    interrupted = False
    t = len(classifiers) * len(datasets) * len(seeds) * len(trainpoints) * len(expansions)
    k = 0
    #print("Total number of steps: " + str(t))
    pbar = tqdm(total=t)
    itsSinceLastSave = 0
    basefeaturesStr = str(basefeatures).replace(",", ";")
    for ds in datasets:
        for c in classifiers:
            for exp in expansions:
                for tp in trainpoints:
                    observedFail = False
                    for seed in seeds:
                        try:
                            k += 1
                            pbar.update(1)
                            if observedFail:
                                continue
                            #print("Learning model for " + str(ds) + "/" + c + " with seed " + str(seed) + " on " + str(tp) + " and expansions " + str(exp) + " (" + str(np.round(100 * k/t, 2)) + "%)")
                            
                            excerpt = serializedDF.query("openmlid == " + str(ds) + " and algorithm == '" + c + "' and learner == '" + learner + "' and trainpoints_learner == " + str(tp) + " and basefeatures == \"" + basefeaturesStr + "\" and expansions == '" + str(exp) + "' and seed == " + str(seed))
                            exists = len(excerpt) > 0
                            
                            if exists:
                                pass#print("Skipping due to existence")
                            else:
                                train_results = runExperiment(mfdf, basefeatures, ds, c, learner, tp, exp, "traintime", sampleSizes)
                                prediction_results = runExperiment(mfdf, basefeatures, ds, c, learner, tp, exp, "predictiontimeperinstance", sampleSizes)
                 
                                # update the dataframe
                                #if exists:
                                #    serializedDF.loc[excerpt.query("recordtype == 'traintime_gt'").index.values[0]] = [ds, c, seed, learner, tp, str(exp), "traintime_gt", implode(train_results[0], ";")]
                                #    serializedDF.loc[excerpt.query("recordtype == 'traintime_pr'").index.values[0]] = [ds, c, seed, learner, tp, str(exp), "traintime_pr", implode(train_results[1], ";")]
                                #    serializedDF.loc[excerpt.query("recordtype == 'validationtime_gt'").index.values[0]] = [ds, c, seed, learner, tp, str(exp), "validationtime_gt", implode(prediction_results[0], ";")]
                                #    serializedDF.loc[excerpt.query("recordtype == 'validationtime_pr'").index.values[0]] = [ds, c, seed, learner, tp, str(exp), "validationtime_pr", implode(prediction_results[1], ";")]
                                #else:
                                extDF = pd.DataFrame([], columns = serializedDF.columns)
                                for sampleSize in train_results:
                                    if len(train_results[sampleSize][0]) > 0:
                                        if len(train_results[sampleSize][0]) != len(train_results[sampleSize][1]):
                                            raise Exception("Length of GT and PR does not coincide!")
                                        extDF.loc[len(extDF)] = [ds, c, seed, learner, sampleSize, tp, basefeaturesStr, str(exp), "traintime_gt", implode(train_results[sampleSize][0], ";")]
                                        extDF.loc[len(extDF)] = [ds, c, seed, learner, sampleSize, tp, basefeaturesStr, str(exp), "traintime_pr", implode(train_results[sampleSize][1], ";")]
                                for sampleSize in prediction_results:
                                    if len(prediction_results[sampleSize][0]) > 0:
                                        if len(prediction_results[sampleSize][0]) != len(prediction_results[sampleSize][1]):
                                            raise Exception("Length of GT and PR does not coincide!")
                                        extDF.loc[len(extDF)] = [ds, c, seed, learner, sampleSize, tp, basefeaturesStr, str(exp), "predictiontime_gt", implode(prediction_results[sampleSize][0], ";")]
                                        extDF.loc[len(extDF)] = [ds, c, seed, learner, sampleSize, tp, basefeaturesStr, str(exp), "predictiontime_pr", implode(prediction_results[sampleSize][1], ";")]
                                serializedDF = pd.concat([serializedDF, extDF])
                                
                                # update csv file
                                if itsSinceLastSave >= 50:
                                    serializedDF.to_csv("runtime_predictions.csv", index=False)
                                    itsSinceLastSave = 0
                                else:
                                    itsSinceLastSave += 1 
                        except KeyboardInterrupt:
                            print("Interrupted")
                            interrupted = True
                            observedFail = True
                            break
                        except Exception as ex:
                            if "classifier space has no values" in ex.args[0] or "No test data selected" in ex.args[0]:
                                print(ex)
                                observedFail = True
                                pass
                            else:
                                raise
                if interrupted:
                    break
            if interrupted:
                break
        if interrupted:
            break
    pbar.close()
    serializedDF.to_csv("runtime_predictions.csv", index=False)

def runAllBaseAlgorithmExperiments(exp, dataspace, learner):
    conducted = 0

    # invoke each experiment with the data of the data space
    for row_id, e in enumerate(exp.values):
        evalId = e[0]
        classifier = e[2]
        mse_trainruntime_half_index = np.where(exp.columns == ("mse_" + learner + "_trainruntime_half"))[0][0]
        mse_trainruntime_full_index = np.where(exp.columns == ("mse_" + learner + "_trainruntime_full"))[0][0]
        mse_testruntime_half_index = np.where(exp.columns == ("mse_" + learner + "_testruntime_half"))[0][0]
        mse_testruntime_full_index = np.where(exp.columns == ("mse_" + learner + "_testruntime_full"))[0][0]
        if e[mse_trainruntime_half_index]  != -1:
            print("Skipping experiment for which already data is available.")
            continue
        print("Conducting " + str(conducted + 1) + "-th open experiment " + str(evalId) + ". Dataset " + str(e[1]) + ", classifier: " + classifier + ". " + str(e[3]) + " training instances, seed " + str(e[4]) + ", expansions: " + str(e[5]) + ". Using learner: " + learner)
        erHalf, erFull = runExperiment(dataspace, e[1], classifier, learner, e[3], e[4], e[5], "traintime")
        exp.iat[row_id, mse_trainruntime_half_index] = erHalf
        exp.iat[row_id, mse_trainruntime_full_index] = erFull
        erHalf, erFull = runExperiment(dataspace, e[1], classifier, learner, e[3], e[4], e[5], "testtime")
        exp.iat[row_id, mse_testruntime_half_index] = erHalf
        exp.iat[row_id, mse_testruntime_full_index] = erFull
        conducted += 1
        
        
        



