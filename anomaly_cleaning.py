from pyod.models.auto_encoder import AutoEncoder
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.lmdd import LMDD
from pyod.models.cblof import CBLOF
from pyod.models.knn import KNN
import numpy as np

def cleanAnomalies(X,y,perc=0.01):
    #detector = AutoEncoder(hidden_neurons=[256,128,64,32,16,16,32,64,128,256], epochs=50).fit(X)
    #detector = HBOS().fit(X)
    #detector = IForest(n_estimators=300, bootstrap=True, n_jobs=8, random_state=123456789, verbose=1).fit(X)
    #detector = LMDD(n_iter=50).fit(X) # Muy lento
    #detector = CBLOF(n_clusters=3, n_jobs=8).fit(X)
    detector = KNN(n_neighbors=7, n_jobs=20).fit(X)
    sorted = np.argsort(detector.decision_scores_)[::-1]
    size = len(X)-int(perc*len(X))
    return X[sorted[:size]], y[sorted[:size]]
