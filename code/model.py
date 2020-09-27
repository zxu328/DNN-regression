import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class DLRegresser(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(DLRegresser, self).__init__()
      
        self.fc1 = nn.Linear(D_in, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, H1) # symetric structure
        self.fc4 = nn.Linear(H1, D_out)
        self.bn1 = nn.BatchNorm1d(D_in)
        self.act = nn.Sigmoid()
        self.dout = nn.Dropout(0.2)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.bn1(x)
        x = self.fc1(x)
        
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.dout(x)
        x = self.fc2(x)
        x = F.leaky_relu(x, negative_slope=0.2)
    #    x = self.dout(x)
        x = self.fc3(x)
        x = F.leaky_relu(x, negative_slope=0.2)
    #    x = self.dout(x)
        x = self.fc4(x)
        return x
    