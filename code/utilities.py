# Gun owning project
# utilities functions

import numpy as np
import pandas as pd

def loadData(fileName):
    data = pd.read_excel(fileName)
    state = data.ST
    GunOwnership = data.GO.to_numpy()
    FsToS = data.FS.to_numpy()
    HuntingLic = data.HL.to_numpy()
    
    
    v = 2
    n = GunOwnership.size
    
    # forming input matrix and output array
    X = np.zeros((v, n))
    X[0, :] = FsToS
    X[1, :] = HuntingLic
    
    y = GunOwnership
    
    return X, y

def calRSquare(y, y_hat):
    y_mean = np.mean(y)
    errorSq = (y - y_hat) ** 2
    SSE = np.sum(errorSq)
    errorMean = (y - y_mean) ** 2
    SST = np.sum(errorMean)
    RSq = 1- SSE / SST
    return RSq



