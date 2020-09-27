import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import model as M
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

class DNNClass:
    def __init__(self, l2Reg, lr, epochs, H1, H2, TrainCri, ValCri, device, X_test2, y_test2, reload):
        self.reload = reload
        if reload:
            self.model = torch.load('./savedModel.pt')
            self.model.cuda(device)
            self.epochs = 0
        else:
            self.model = M.DLRegresser(2, H1, H2, 1).cuda(device)
            self.epochs = epochs
        self.l2_reg = l2Reg
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = self.l2_reg)
        
        self.TrainCri = TrainCri
        self.ValCri = ValCri
        self.device = device
        self.X_test2 = X_test2
        self.y_test2 = y_test2
        
        
        
    def fit(self, XTrain, yTrain, XVal, yVal, test2 = 1):
        yTrainMean = np.mean(yTrain) * np.ones(yTrain.shape)
        yValMean = np.mean(yVal) * np.ones(yVal.shape)
        MSTTrain = mean_squared_error(yTrain, yTrainMean)
        MSTVal = mean_squared_error(yVal, yValMean)
        NTrain = yTrain.size
        NVal = yVal.size
        xTrain = torch.from_numpy(XTrain).float().to(self.device)
        xVal = torch.from_numpy(XVal).float().to(self.device)
        yTrain = torch.from_numpy(yTrain).float()
        yVal = torch.from_numpy(yVal).float()
        yTrain = yTrain.view(NTrain, 1).to(self.device)
        yVal = yVal.view(NVal, 1).to(self.device)
        if test2 == 1:
            yTestMean = np.mean(self.y_test2) * np.ones(self.y_test2.shape)
            MSTTest2 = mean_squared_error(self.y_test2, yTestMean)
            NTest2 = self.y_test2.size
            xTest2 = torch.from_numpy(self.X_test2).float().to(self.device)
            yTest2 = torch.from_numpy(self.y_test2).float().to(self.device)
            yTest2 = yTest2.view(NTest2, 1).to(self.device)
            
        
        loss_fn = torch.nn.MSELoss()
        if self.reload:
            a = 1.0
            b = 1.0
            c = 1.0
            yPredict = self.model(xTrain) 
            yValPredict = self.model(xVal)
            yTest2Predict = self.model(xTest2)
            yPredict = yPredict.cpu()
            yValPredict = yValPredict.cpu()
            yTest2Predict = yTest2Predict.cpu()
            yPredict = yPredict.data.numpy()
            yValPredict = yValPredict.data.numpy()
            yTest2Predict = yTest2Predict.data.numpy()
            return a, b, c, yPredict, yValPredict, yTest2Predict
            
            
        
        for t in range(self.epochs):
            yPredict = self.model(xTrain)
            loss = loss_fn(yPredict, yTrain)
            yValPredict = self.model(xVal)
            lossVal = loss_fn(yValPredict, yVal)
            if test2 == 1:
                yTest2Predict = self.model(xTest2)
                lossTest2 = loss_fn(yTest2Predict, yTest2)
                c = 1.0 - lossTest2.cpu().data.numpy() / MSTTest2
            
            
            
    
            a = 1.0 - loss.cpu().data.numpy() / MSTTrain 
            b = 1.0 - lossVal.cpu().data.numpy() / MSTVal
            
            if test2 == 1:
                if (a > self.TrainCri and b > self.ValCri): 
                    print(loss.item(), 'train:', a)
                    print(lossVal.item(), 'test:', b)
                    
                    yPredict = yPredict.cpu()
                    yValPredict = yValPredict.cpu()
                    yTest2Predict = yTest2Predict.cpu()
                    yPredict = yPredict.data.numpy()
                    yValPredict = yValPredict.data.numpy()
                    yTest2Predict = yTest2Predict.data.numpy()
                    return a, b, c, yPredict, yValPredict, yTest2Predict
                
            else:
    
                if (a > self.TrainCri and b > self.ValCri):
    
                    print(loss.item(), 'train:', a)
                    print(lossVal.item(), 'test:', b)
                    
                    return a, b
    
            if (t % 100 == 0):
                if test2 == 1:
                    print('iterationK', t, 'loss:', loss.item(), 'valLoss', lossVal.item())
                    print('a:', a, 'b:', b)
                    
                else:
                    print('iterationK', t, 'loss:', loss.item(), 'valLoss', lossVal.item())
                    print('a:', a, 'b:', b)
                
                
            self.model.zero_grad()

            loss.backward()

            self.optimizer.step()
            
        if test2 == 1:    
            return a, b, c, yPredict, yValPredict, yTest2Predict
        else:
            return a, b
        
        
    def predict(self, X):
        X = torch.from_numpy(X).float()
        X = X.to(self.device)
        yPredict = self.model(X)
        yPredict = yPredict.cpu()
        yPredict = yPredict.data.numpy()
        
        return yPredict
    
    def saveModel(self):
        torch.save(self.model, './savedModel.pt')
        
        
        
        
        
        
