# KFold CV DNN
import numpy as np
import utilities as ut
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from DNNClass import DNNClass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def calRSquare(y, yHat):
    RSquare = r2_score(y, yHat)
    return RSquare
def calTestRSquare(rgs, X, y):
    yHat = rgs.predict(X)
    RSquare = calRSquare(y, yHat)
    return RSquare
    
def cvEstimateDNN(nSplit, rgs, X_train, y_train):
    cv = KFold(n_splits=nSplit)
    valScore = 0.0
    trainScore = 0.0
    for train, val in cv.split(X_train, y_train):
        a, b = rgs.fit(X_train[train], y_train[train], X_train[val], y_train[val], test2 = 0)
        trainScore += a
        valScore += b
        print(a, b)
    trainScore /= nSplit
    valScore /= nSplit
    print(trainScore)
    print(valScore)
    
    return trainScore, valScore

# Load data
    
torch.cuda.empty_cache()
device = torch.device("cuda:0")

fileNameTrain = './TrainingData.xlsx'
fileNameTest = './TestingData.xlsx'
fileNameTest2 = './TestingData2.xlsx'
X_train, y_train = ut.loadData(fileNameTrain)
X_test, y_test = ut.loadData(fileNameTest)
X_test2, y_test2 = ut.loadData(fileNameTest2)
X_train = X_train.T
X_test = X_test.T
X_test2 = X_test2.T

reload = 1
needCV = 0

DNN = DNNClass(l2Reg = 0.0, lr = 1e-5, epochs = 100000, H1 = 500, H2 = 1000, TrainCri = 0.95, ValCri = 0.95, device = device, X_test2 = X_test2, y_test2 = y_test2, reload = reload)
if (not reload) and needCV:
    DNNTrainRSquare, DNNValRSquare = cvEstimateDNN(10, DNN, X_train, y_train)

#_, DNNTestSquare, DNNTest2Square, y_hatTrain, y_hatTest, y_hatTest2 = DNN.fit(X_train, y_train, X_test, y_test, test2 = 1)
_, DNNTestSquare, DNNTest2Square, y_hatTrain, y_hatTest, y_hatTest2 = DNN.fit(X_train, y_train, X_test, y_test, test2 = 1)

saveModel = 0
if saveModel:
    DNN.saveModel()

if (not reload) and needCV:
    print(DNNValRSquare, DNNTestSquare)

# output residual
out = 1

if out:

    fidTrain = open('./DNNResidualTrain.dat', 'w')
    fidTest = open('./DNNResidualTest.dat', 'w')
    fidTest2 = open('./DNNResidualTest2.dat', 'w')
    
    
    for i in range(50):
        fidTrain.write('%e \n' % ((y_train[i] - y_hatTrain[i])))
        
    for i in range(49):
        fidTest.write('%e \n' % ((y_test[i] - y_hatTest[i])))
        
    for i in range(14):
        fidTest2.write('%e \n' % ((y_test2[i] - y_hatTest2[i])))
        
        
    fidTrain.close()
    fidTest.close()   
    fidTest2.close()     

    

