import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from .networkUtil import *

class dilatedConvBlock(nn.Module):
    def __init__(self, iConvNum = 3):
        super(dilatedConvBlock, self).__init__()
        self.LRelu = nn.LeakyReLU()
        convList = []
        for i in range(1, iConvNum+1):
            tmpConv = nn.Conv2d(32,32,3,padding = i, dilation = i)
            convList.append(tmpConv)
        self.layerList = nn.ModuleList(convList)
    
    def forward(self, x1):
        x2 = x1
        for conv in self.layerList:
            x2 = conv(x2)
            x2 = self.LRelu(x2)
        
        return x2
    
class RDN_recursionUnit(nn.Module):
    def __init__(self, convNum = 3, recursiveTime = 3, inChannel = 2):
        super(RDN_recursionUnit, self).__init__()
        self.rTime = recursiveTime
        self.LRelu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(inChannel,32,3,padding = 1)
        self.dilateBlock = dilatedConvBlock(convNum)
        self.conv2 = nn.Conv2d(32,2,3,padding = 1)
        
    def forward(self, x1):
        x2 = self.conv1(x1)
        x2 = self.LRelu(x2)
        xt = x2
        for i in range(self.rTime):
            x3 = self.dilateBlock(xt)
            xt = x3+x2
        x4 = self.conv2(xt)
        x4 = self.LRelu(x4)
        x5 = x4+x1
        
        return x5
    
class dataFidelityUnit(nn.Module):
    def __init__(self, initLamda = 10e6):
        super(dataFidelityUnit, self).__init__()
        self.normalized = True
        self.lamda = initLamda
    
    def forward(self, xin, y, mask):
        # y: aliased image 
        # x1: reconstructed image
        # mask: sampling mask
        mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],1)
        
        xin_c = xin
        xin_c = xin_c.permute(0,2,3,1)
        
        xin_f = torch.fft(xin_c,2, normalized=self.normalized)
        xGT_f = y
        
        xout_f = xin_f + (- xin_f + xGT_f) * iScale * mask

        xout = torch.ifft(xout_f,2, normalized=self.normalized)
        xout = xout.permute(0,3,1,2)
        
        return xout

class RDN_complex(nn.Module):
    def __init__(self, b=5, d=3, r=3, xin1 = 2, dcLayer = 'DF'):
        super(RDN_complex, self).__init__()
        templayerList = []
        for i in range(b):
            if(i==0):
                tmpConv = RDN_recursionUnit(d,r,inChannel=xin1)
            else:
                tmpConv = RDN_recursionUnit(d,r)
            if(dcLayer=='DF'):
                tmpDF = dataFidelityUnit()
            else:
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
        
    
