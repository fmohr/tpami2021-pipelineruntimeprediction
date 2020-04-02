from libruntimelearning import *

def RMSE(A, F):
    return np.sqrt(sk.metrics.mean_squared_error(A, F))

def SMAPE(A, F):
    n = len(A)
    s = 0
    for i in range(n):
        s += 2 * np.abs(F[i] - A[i]) / (F[i] + A[i])
    return s / n

def overMAPE(A, F):
    n = 0
    s = 0
    for i in range(len(A)):
        if A[i] <= F[i]:
            s += (F[i] - A[i]) / A[i]
            n+=1
    if n == 0:
        return 0
    return s / n

def underMAPE(A, F):
    n = 0
    s = 0
    for i in range(len(A)):
        if A[i] >= F[i]:
            s += (A[i] - F[i]) / F[i]
            n+=1
    if n == 0:
        return 0
    return s / n

def RDIST(A, F):
    n = len(A)
    s = 0
    for i in range(n):
        s += np.abs(F[i] - A[i]) / np.min([F[i], A[i]])
    return s / n

def getMeasureForDatasetClassifierPairs(datasets, classifiers, data, measure):
    output = np.zeros((len(datasets), len(classifiers)))
    for i, d in enumerate(datasets):
        for j, c in enumerate(classifiers):
            if type(data[i][j]) == list and len(data[i][j]) > 0:
                output[i][j] = measure(data[i][j][0][0], data[i][j][0][1])
    return output

def getAllMetrics(datasets, classifiers, data_train, data_pred):
    metrics = {}
    metrics['RMSE_train'] = getMeasureForDatasetClassifierPairs(datasets, classifiers, data_train, RMSE)
    metrics['RMSE_test'] = getMeasureForDatasetClassifierPairs(datasets, classifiers, data_pred, RMSE)
    return metrics

def getAllMetricsForLearnerFromFile(learner):
    datasets, algorithms, trainSeries, predictionSeries = getGTPredictionPairSeries(learner)
    return getAllMetrics(datasets, algorithms, trainSeries, predictionSeries)