from pyod.models.knn import KNN
import numpy as np

def cleanAnomalies(X,y,perc=0.01):
    detector = KNN(n_neighbors=5, n_jobs=20).fit(X)
    sorted = np.argsort(detector.decision_scores_)[::-1]
    size = len(X)-int(perc*len(X))
    return X[sorted[:size]], y[sorted[:size]]
