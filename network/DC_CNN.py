import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .networkUtil import *

class convBlock(nn.Module):
    def __init__(self, iConvNum = 5, f=64):
        super(convBlock, self).__init__()
        self.Relu = nn.ReLU()
        self.conv1 = nn.Conv2d(2,f,3,padding = 1)
        convList = []
        for i in range(1, iConvNum-1):
            tmpConv = nn.Conv2d(f,f,3,padding = 1)
            convList.append(tmpConv)
        self.layerList = nn.ModuleList(convList)
        self.conv2 = nn.Conv2d(f,2,3,padding = 1)
    
    def forward(self, x1):
        x2 = self.conv1(x1)
        x2 = self.Relu(x2)
        for conv in self.layerList:
            x2 = conv(x2)
            x2 = self.Relu(x2)
        x3 = self.conv2(x2)
        
        x4 = x3 + x1
        
        return x4

class DC_CNN(nn.Module):
    def __init__(self, d = 5, c = 5, fNum = 32):
        super(DC_CNN, self).__init__()
        templayerList = []
        for i in range(c):
            tmpConv = convBlock(d, fNum)
            tmpDF = dataConsistencyLayer_static()
            templayerList.append(tmpConv)
            templayerList.append(tmpDF)
        self.layerList = nn.ModuleList(templayerList)
        
    def forward(self, x1, y, mask):
        xt = x1
        flag = True
        for layer in self.layerList:
            if(flag):
                xt = layer(xt)
                flag = False
            else:
                xt = layer(xt, y, mask)
                flag = True
        
        return xt