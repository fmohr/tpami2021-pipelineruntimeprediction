import pandas as pd
import numpy as np
import sklearn as sk

def RMSE(A, F):
    if type(A) == list:
        A = np.array(A)
    elif type(A) == pd.core.series.Series:
        A = np.array([A.iloc[i] for i in range(len(A))])
    if type(A) != np.ndarray:
        raise Exception("Unsupported types.")
        
    if type(F) == list:
        F = np.array(F)
    elif type(F) == pd.core.series.Series:
        F = np.array([F.iloc[i] for i in range(len(F))])
    if type(F) != np.ndarray:
        raise Exception("Unsupported types.")
    
    if A.shape != F.shape:
        raise Exception("Expected and actual have different dimensions!")
    
    if len(A.shape) <= 1:
        return np.sqrt(sk.metrics.mean_squared_error(np.nan_to_num(A), np.nan_to_num(F)))
    else:
        return [np.sqrt(sk.metrics.mean_squared_error(np.nan_to_num(A[i]), np.nan_to_num(F[i]))) for i in range(A.shape[0])]
    
def MAPE(A, F):
    if type(A) == list:
        A = np.array(A)
    elif type(A) == pd.core.series.Series:
        A = np.array([A.iloc[i] for i in range(len(A))])
    if type(A) != np.ndarray:
        raise Exception("Unsupported types.")
        
    if type(F) == list:
        F = np.array(F)
    elif type(F) == pd.core.series.Series:
        F = np.array([F.iloc[i] for i in range(len(F))])
    if type(F) != np.ndarray:
        raise Exception("Unsupported types.")
    
    if A.shape != F.shape:
        raise Exception("Expected and actual have different dimensions!")
    
    if len(A.shape) <= 1:
        n = 0
        s = 0
        for i in range(len(A)):
            s += np.abs((A[i] - F[i]) / A[i])
            n+=1
        return s / n if n > 0 else 0
    return [MAPE(A[i], F[i]) for i in range(len(A))]

def RMSEOverX(A, F, x):
    indices = np.where(np.maximum(A, F) > x)[0]
    return RMSE(A[indices], F[indices]) if len(indices) > 0 else np.nan

def RMSEOverOne(A, F):
    return RMSEOverX(A, F, 1)

def RMSEOverTwo(A, F):
    return RMSEOverX(A, F, 2)

def RMSEOverTen(A, F):
    return RMSEOverX(A, F, 10)

def overMAPE(A, F):
    if type(A) == list:
        A = np.array(A)
    elif type(A) == pd.core.series.Series:
        A = np.array([A.iloc[i] for i in range(len(A))])
    if type(A) != np.ndarray:
        raise Exception("Unsupported types.")
        
    if type(F) == list:
        F = np.array(F)
    elif type(F) == pd.core.series.Series:
        F = np.array([F.iloc[i] for i in range(len(F))])
    if type(F) != np.ndarray:
        raise Exception("Unsupported types.")
    
    if A.shape != F.shape:
        raise Exception("Expected and actual have different dimensions!")
    
    if len(A.shape) <= 1:
        n = 0
        s = 0
        for i in range(len(A)):
            if A[i] <= F[i] and (F[i] > 10):
                s += (F[i] - A[i]) / np.maximum(1, A[i])
                n+=1
        return s / n if n > 0 else 0
    return [overMAPE(A[i], F[i]) for i in range(len(A))]

def underMAPE(A, F):
    if type(A) == list:
        A = np.array(A)
    elif type(A) == pd.core.series.Series:
        A = np.array([A.iloc[i] for i in range(len(A))])
    if type(A) != np.ndarray:
        raise Exception("Unsupported types.")
        
    if type(F) == list:
        F = np.array(F)
    elif type(F) == pd.core.series.Series:
        F = np.array([F.iloc[i] for i in range(len(F))])
    if type(F) != np.ndarray:
        raise Exception("Unsupported types.")
    
    if A.shape != F.shape:
        raise Exception("Expected and actual have different dimensions!")
        
    if len(A.shape) <= 1:
        n = 0
        s = 0
        for i in range(len(A)):
            if A[i] >= F[i] and (A[i] > 10):
                s += (A[i] - F[i]) / np.maximum(1, A[i])
                n+=1
        return s / n if n > 0 else 0
    raise Exception("Uncovered case")

def getAllMetrics(datasets, classifiers, data_train, data_pred, predicate):
    metrics = {}
    for measure in [RMSE, SMAPE, RDIST, overMAPE, underMAPE]:
        measureName = str(measure.__name__)
        metrics[measureName] = {}
        metrics[measureName]['train'] = getMeasureForDatasetClassifierPairs(datasets, classifiers, data_train, measure, predicate)
        metrics[measureName]['test'] = getMeasureForDatasetClassifierPairs(datasets, classifiers, data_pred, measure, predicate)
    return metrics

def createPredictionErrorFile(suffix = "", query = None):
    inFile = "data/workdata/runtimepredictions" + suffix + ".csv"
    outFile = "data/workdata/runtimepredictions" + suffix + "_withmetrics.csv"
    serializedDF = pd.read_csv(inFile)
    if not query is None:
        serializedDF = serializedDF.query(query)
    dfOut = pd.read_csv(outFile)
    attributes = ["openmlid", "algorithm", "learner", "trainpoints_algorithm", "trainpoints_learner", "basefeatures", "expansions"]
    groups = serializedDF.groupby(attributes)
    measures = [RMSE, RMSEOverOne, RMSEOverTwo, RMSEOverTen, SMAPE, overMAPE, underMAPE]
    resultCols = []
    for measure in measures:
        name = str(measure.__name__)
        resultCols.append("fittime_" + name)
        resultCols.append("predicttime_" + name)
        
    cols = attributes.copy()
    cols.extend(resultCols)
    rows = []
    i = 0
    for gIndex, group in tqdm(groups):
        print(gIndex)
        row = list(gIndex)
        matches = dfOut[np.count_nonzero((dfOut[attributes] == row).values, axis=1) == len(attributes)]
        if len(matches) == 0:

            # sub-group by seed
            subresults = {}
            for rc in resultCols:
                subresults[rc] = []
            for sgIndex, subgroup in group.groupby("seed"):

                if len(subgroup) >= 4:
                    print("seed: ", sgIndex)

                    # compute basis to create metrics
                    trainGT = np.array(str(subgroup.query("recordtype == 'traintime_gt'")["entries"].values[0]).split(";")).astype('float64')
                    trainPR = np.array(str(subgroup.query("recordtype == 'traintime_pr'")["entries"].values[0]).split(";")).astype('float64')
                    testGT = np.array(str(subgroup.query("recordtype == 'predictiontime_gt'")["entries"].values[0]).split(";")).astype('float64')
                    testPR = np.array(str(subgroup.query("recordtype == 'predictiontime_pr'")["entries"].values[0]).split(";")).astype('float64')
                    if len(trainGT) != len(trainPR):
                        raise Exception("GT and PR of traintimes have different length!")
                    if len(testGT) != len(testPR):
                        raise Exception("GT and PR of prediction times have different length!")

                    # compute metrics
                    for measure in measures:
                        measureName = str(measure.__name__)
                        print("Measure " + measureName)
                        subresults["fittime_" + measureName].append(measure(trainGT, trainPR))
                        subresults["predicttime_" + measureName].append(measure(testGT, testPR))
                        #print(measureName + ": " + str(subresults["fittime_" + measureName][-1]) + "/" + str(subresults["predicttime_" + measureName][-1]))
                        
                    #if np.abs(np.mean(testPR)-np.mean(testGT) > 30):
                    #    print(gIndex)
                    #    print(testPR)
                    #    print(testGT)

            for sr in subresults:
                row.append(np.mean(subresults[sr]))
            i += 1
        
            rows.append(row)
            if i % 1000 == 0:
                df = pd.concat([dfOut, pd.DataFrame(rows, columns = cols)])
                df.to_csv(outFile, index=False)
                i = 0
                
    if i > 0:
        df = pd.concat([dfOut, pd.DataFrame(rows, columns = cols)])
        df.to_csv(outFile, index=False)
    else:
        df = dfOut
    return df

def getMeasureForDatasetClassifierPairs(datasets, classifiers, data, measure, predicate):
    output = np.zeros((len(datasets), len(classifiers)))
    for i, d in enumerate(datasets):
        for j, c in enumerate(classifiers):
            if type(data[i][j]) == list and len(data[i][j]) > 0:
                A = np.nan_to_num(data[i][j][0][0])
                F = np.nan_to_num(data[i][j][0][1])
                
                if predicate is None:
                    indices = np.arange(len(A))
                else:
                    indices = np.where(predicate(A, F))[0] 
                output[i][j] = measure(A[indices], F[indices]) if len(indices) > 0 else np.nan
    return output