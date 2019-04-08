import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .networkUtil import *
from .DDU import DilatedDenseUnet

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
        #x3 = self.Relu(x3)
        
        x4 = x3 + x1
        
        return x4
    
# class dataConsistencyLayer_static(nn.Module):
#     def __init__(self, initLamda = 1):
#         super(dataConsistencyLayer_static, self).__init__()
#         self.normalized = True #norm == 'ortho'
# #         self.lamda = Parameter(torch.Tensor(1))
# #         self.lamda.data.uniform_(0, 1)

    
#     def forward(self, xin, y, mask):
#         #iScale = self.lamda/(1+self.lamda)
#         iScale = 1
#         mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],1)
        
#         xin_c = xin.permute(0,2,3,1)
#         xGT_c = y.permute(0,2,3,1)
        
#         xin_f = torch.fft(xin_c,2, normalized=self.normalized)
#         xGT_f = torch.fft(xGT_c,2, normalized=self.normalized)
        
#         xout_f = xin_f + (- xin_f + xGT_f) * iScale * mask

#         xout = torch.ifft(xout_f,2, normalized=self.normalized)
#         xout = xout.permute(0,3,1,2)
        
#         return xout

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
        #y = x1
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


class DC_CNN_DDU(nn.Module):
    def __init__(self, c = 2):
        super(DC_CNN_DDU, self).__init__()
        templayerList = []
        for i in range(c):
            tmpConv = DilatedDenseUnet(1,isComplex=True)
            tmpDF = dataConsistencyLayer_static()
            templayerList.append(tmpConv)
            templayerList.append(tmpDF)
        self.layerList = nn.ModuleList(templayerList)
        
    def forward(self, x1, y, mask):
        #y = x1
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