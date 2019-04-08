import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .networkUtil import *   

class DC_CNN_dynamic_DS(nn.Module):
    def __init__(self, d = 5, c = 10, nadj = 5):
        super(DC_CNN_dynamic_DS, self).__init__()
        self.nadj = nadj
        
        templayerList = []
        for i in range(1,nadj+1):
            tmpDS = dataSharing(i)
            templayerList.append(tmpDS)
        self.DSList = nn.ModuleList(templayerList)
        
        templayerList = []
        for i in range(c):
            tmpConv = convBlock(d,12)
            tmpDF = dataConsistencyLayer()
            templayerList.append(tmpConv)
            templayerList.append(tmpDF)
        self.layerList = nn.ModuleList(templayerList)
        
    def forward(self, x1, mask):
        y = x1[:,0:2]
        xt = x1
        flag = True
        dsFlag = False
        for layer in self.layerList:
            if(flag):
                xt2 = xt
                for ds in self.DSList:
                    xds = ds(xt,mask,dsFlag)
                    xt2 = torch.cat([xt2,xds],1)
                xt = layer(xt2)
                dsFlag = True
                flag = False
            else:
                xt = layer(xt, y, mask)
                flag = True
        
        return xt

class convBlock(nn.Module):
    def __init__(self, iConvNum = 5, inChannel = 2):
        super(convBlock, self).__init__()
        self.Relu = nn.ReLU()
        self.conv1 = nn.Conv3d(inChannel,64,3,padding = 1)
        convList = []
        for i in range(1, iConvNum):
            tmpConv = nn.Conv3d(64,64,3,padding = 1)
            convList.append(tmpConv)
        self.layerList = nn.ModuleList(convList)
        self.conv2 = nn.Conv3d(64,2,3,padding = 1)
    
    def forward(self, x1):
        x2 = self.conv1(x1)
        x2 = self.Relu(x2)
        for conv in self.layerList:
            x2 = conv(x2)
            x2 = self.Relu(x2)
        x3 = self.conv2(x2)
        # x3 = self.Relu(x3)
        
        x4 = x3 + x1[:,0:2]
        
        return x4
    
class dataConsistencyLayer(nn.Module):
    def __init__(self, initLamda = 1):
        super(dataConsistencyLayer, self).__init__()
        self.lamda = Parameter(torch.Tensor(1))
        self.lamda.data.uniform_(0, 1)
        #self.lamda.data = torch.Tensor([initLamda])
        #self.lamda = initLamda
    
    def forward(self, xin, y, mask):
        iScale = self.lamda/(1+self.lamda)
        mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],mask.shape[3],1)
        
        xin_c = xin.permute(0,2,3,4,1)
        xGT_c = y.permute(0,2,3,4,1)
        
        xin_f = torch.fft(xin_c,2)
        xGT_f = torch.fft(xGT_c,2)
        
        xout_f = xin_f + (- xin_f + xGT_f) * iScale * mask

        xout = torch.ifft(xout_f,2)
        xout = xout.permute(0,4,1,2,3)
        
        return xout
    
class dataSharing(nn.Module):
    def __init__(self, nadj=1):
        super(dataSharing, self).__init__()
        self.nadj = nadj
        
    def forward(self,x1,mask,fullSharingMask=False):
#         if(self.nadj == 0):
#             return x1
        mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],mask.shape[3],1)
        x1_c = x1.permute(0,2,3,4,1)
        x1_f = torch.fft(x1_c,2)
        
        xfake = torch.zeros((x1_f.shape[0],x1_f.shape[1]+self.nadj*2,x1_f.shape[2],x1_f.shape[3],2)).cuda()
        xfake[:,self.nadj:-self.nadj] = x1_f
        mfake = torch.zeros(mask.shape[0],mask.shape[1]+self.nadj*2,mask.shape[2],mask.shape[3],1).cuda()
        if(fullSharingMask):
            mfake[:,self.nadj:-self.nadj] = 1
        mfake[:,self.nadj:-self.nadj] = mask
        
        rawTimeSlice = x1_f.shape[1]
        xshare = torch.zeros_like(x1_f)
        mshare = torch.zeros_like(mask)
        for i in range(self.nadj):
            xshare += xfake[:,i:-self.nadj*2+i]*mfake[:,i:-2*self.nadj+i]
            xshare += xfake[:,self.nadj*2-i:self.nadj*2-i+rawTimeSlice]*mfake[:,self.nadj*2-i:self.nadj*2-i+rawTimeSlice]
            mshare += mfake[:,i:-2*self.nadj+i]
            mshare += mfake[:,self.nadj*2-i:self.nadj*2-i+rawTimeSlice]
        
        mshare = mshare * (1-mask)
        
        xout_f = x1_f
        xout_f[mshare[:,:,:,:,0]!=0] = xshare[mshare[:,:,:,:,0]!=0]/mshare[mshare[:,:,:,:,0]!=0]
        
        xout = torch.ifft(xout_f,2)
        xout = xout.permute(0,4,1,2,3)
        
        return xout
        

class DC_CNN_dynamic(nn.Module):
    def __init__(self, d = 5, c = 10):
        super(DC_CNN_dynamic, self).__init__()
        templayerList = []
        for i in range(c):
            tmpConv = convBlock(d)
            tmpDF = dataConsistencyLayer()
            templayerList.append(tmpConv)
            templayerList.append(tmpDF)
        self.layerList = nn.ModuleList(templayerList)
        
    def forward(self, x1, mask):
        y = x1[:,0:2]
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