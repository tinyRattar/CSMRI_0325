import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .networkUtil import *

class DDU_ori_withDC(nn.Module):
    def __init__(self, inChannel = 3, activ = 'ReLU', fNum = 16):
        super(DDU_ori_withDC, self).__init__()
        self.DDU = DilatedDenseUnet_ori(inChannel, activ, fNum)
        self.DC = dataConsistencyLayer()
    
    def forward(self, x0, mask):
        x1 = self.DDU(x0)
        x2 = self.DC(x1, x0, mask)
        
        return x2, x1

class DDU_ori(nn.Module):
    def __init__(self, inChannel = 3, activ = 'ReLU', fNum=16):
        super(DDU_ori, self).__init__()
        self.DDU = DilatedDenseUnet_ori(inChannel, activ, fNum)

    def forward(self,x0):
        x1 = self.DDU(x0)

        return x1
        
class dataConsistencyLayer(nn.Module):
    def __init__(self, initLamda = 1):
        super(dataConsistencyLayer, self).__init__()
        self.lamda = Parameter(torch.Tensor(1))
        self.lamda.data.uniform_(0, 1)
        #self.lamda.data = torch.Tensor([initLamda])
        #self.lamda = initLamda
    
    def forward(self, xin, y, mask):
        iScale = self.lamda/(1+self.lamda)
        if(xin.shape[1]==1):
            emptyImag = torch.zeros_like(xin)
            xin_c = torch.cat([xin,emptyImag],1).permute(0,2,3,1)
            xGT_c = torch.cat([y,emptyImag],1).permute(0,2,3,1)
        else:
            xin_c = xin.permute(0,2,3,1)
            xGT_c = y.permute(0,2,3,1)
        mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],1)
        
        xin_f = torch.fft(xin_c,2)
        xGT_f = torch.fft(xGT_c,2)
        
        xout_f = xin_f + (- xin_f + xGT_f) * iScale * mask

        xout = torch.ifft(xout_f,2)
        xout = xout.permute(0,3,1,2)
        if(xin.shape[1]==1):
            xout = xout[:,0:1]
        
        return xout

class DilatedDenseUnet_ori(nn.Module):
    def __init__(self, inChannel = 3, activation = 'ReLU', fNumber = 16):
        super(DilatedDenseUnet_ori, self).__init__()
        dilate = True
        self.c = inChannel
        
        if(activation == 'LeakyReLU'):
            self.activ = nn.LeakyReLU()
        else:
            self.activ = nn.ReLU()
        
        self.block0 = nn.Sequential()
        self.block0.add_module("bn_0",nn.BatchNorm2d(inChannel))
        self.block0.add_module("relu_0",self.activ)
        self.block0.add_module("conv_0", nn.Conv2d(inChannel,16,3,padding=1))
        #self.block0.add_module("relu_0", nn.ReLU())
        
        self.block1 = nn.Sequential()
        self.block1.add_module("conv_1", denseBlock_dilate(16, 3, growthRate=16, layer=3, bottleneckMulti=3, activ = activation))
        self.block1.add_module("transition_1",transitionLayer(64,0.25, activ = activation))
        
        self.down = nn.Sequential()
        self.down.add_module("maxpool_d", nn.MaxPool2d(kernel_size=2))
        self.down.add_module("conv_d1", denseBlock_dilate(16, 3, growthRate=16, layer=3, bottleneckMulti=3, activ = activation))
        self.down.add_module("transition_2",transitionLayer(64,0.25, activ = activation))
        self.down.add_module("conv_d1p", denseBlock_dilate(16, 3, growthRate=16, layer=3, bottleneckMulti=3, activ = activation))
        self.down.add_module("transition_2p",transitionLayer(64,0.25, activ = activation))

        self.middle = nn.Sequential()
        self.middle.add_module("maxpool_m", nn.MaxPool2d(kernel_size=2))
        self.middle.add_module("conv_m1", denseBlock_dilate(16, 3, growthRate=16,layer=3))
        self.middle.add_module("transition_3",transitionLayer(64,0.25, activ = activation))
        self.middle.add_module("conv_m1p", denseBlock_dilate(16, 3, growthRate=16,layer=3))
        self.middle.add_module("transition_3p",transitionLayer(64,0.25, activ = activation))
        self.middle.add_module("upsampling_m", nn.ConvTranspose2d(16,16,2,2))
        
        self.up = nn.Sequential()
        self.up.add_module("bn_u1",nn.BatchNorm2d(32))
        self.up.add_module("relu_u1",self.activ)
        self.up.add_module("conv_s1", nn.Conv2d(32, 16, 1))
        #self.up.add_module("relu_s1", nn.ReLU())
        self.up.add_module("conv_u1", denseBlock_dilate(16, 3, growthRate=16, layer=3, bottleneckMulti=3, activ = activation))
        self.up.add_module("transition_3",transitionLayer(64,0.25, activ = activation))
        self.up.add_module("conv_u1p", denseBlock_dilate(16, 3, growthRate=16, layer=3, bottleneckMulti=3, activ = activation))
        self.up.add_module("transition_3p",transitionLayer(64,0.25, activ = activation))
        self.up.add_module("upsampling_u", nn.ConvTranspose2d(16,16,2,2))
        
        self.block2 = nn.Sequential()
        self.block2.add_module("bn_3",nn.BatchNorm2d(32))
        self.block2.add_module("relu_3",self.activ)
        self.block2.add_module("conv_s2", nn.Conv2d(32, 16, 1))
        #self.block2.add_module("relu_s2", nn.ReLU())
        self.block2.add_module("conv_3", denseBlock_dilate(16, 3, growthRate=16, layer=3, bottleneckMulti=3, activ = activation))
        self.block2.add_module("transition_3",transitionLayer(64,0.25, activ = activation))
        self.block2.add_module("conv_3p", denseBlock_dilate(16, 3, growthRate=16, layer=3, bottleneckMulti=3, activ = activation))
        self.block2.add_module("transition_3p",transitionLayer(64,0.25, activ = activation))
        self.block2.add_module("bn_5",nn.BatchNorm2d(16))
        self.block2.add_module("relu_5",self.activ)
        self.block2.add_module("conv_5", nn.Conv2d(16, 1, 1, padding=0))
        
    def forward(self,x0):
        x1 = self.block0(x0)
        x2 = self.block1(x1)
        x3 = self.down(x2)
        x4 = self.middle(x3)
        
        x5 = torch.cat((x3,x4),1)
        x6 = self.up(x5)
        
        x7 = torch.cat((x2,x6),1)
        x8 = self.block2(x7)
        
        if(self.c == 3):
            result = x8+x0[:,1:2,:,:]
        else:
            result = x8+x0
        
        return result
