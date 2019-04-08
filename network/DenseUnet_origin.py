import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .networkUtil import *

class DUori_cn(nn.Module):
    def __init__(self, inChannel = 3, activ = 'ReLU', dilate = False, fNum=16, c = 2, isComplex = False):
        super(DUori_cn, self).__init__()
        templayerList = []
        for i in range(c):
            tmpBlock = DenseUnet_origin(inChannel, activ, dilate)
            tmpDF = dataConsistencyLayer(isStatic = True)
            templayerList.append(tmpBlock)
            templayerList.append(tmpDF)
        self.layerList = nn.ModuleList(templayerList)

    def forward(self,x0,y,mask):
        xt = x0
        flag = True
        for layer in self.layerList:
            if(flag):
                xt2 = layer(xt)
                flag = False
            else:
                xt = layer(xt2, y, mask)
                flag = True

        return xt,xt2


class DenseUnet_origin(nn.Module):
    def __init__(self, inChannel = 3, activation = 'ReLU', dilate = False):
        super(DenseUnet_origin, self).__init__()
        self.c = inChannel

        if(activation == 'LeakyReLU'):
            self.activ = nn.LeakyReLU()
        elif(activation == 'ReLU'):
            self.activ = nn.ReLU()
        else:
            assert False,"wrong activation type"
        
        self.block0 = nn.Sequential()
        self.block0.add_module("conv_0", nn.Conv2d(inChannel,16,3,padding=1))
        
        self.block1 = nn.Sequential()
        ## WARNING: bottleneckMulti has decreased for debug
        self.block1.add_module("conv_1", denseBlock_origin(16, 3, growthRate=16, layer=3, bottleneckMulti=1,dilationLayer = dilate,activ = activation))
        self.block1.add_module("transition_1",transitionLayer(64,0.25))
        
        self.down = nn.Sequential()
        self.down.add_module("maxpool_d", nn.MaxPool2d(kernel_size=2))
        self.down.add_module("conv_d1", denseBlock_origin(16, 3, growthRate=16, layer=3, bottleneckMulti=1,dilationLayer = dilate,activ = activation))
        self.down.add_module("transition_2",transitionLayer(64,0.25))

        self.middle = nn.Sequential()
        self.middle.add_module("maxpool_m", nn.MaxPool2d(kernel_size=2))
        self.middle.add_module("conv_m1", denseBlock_origin(16, 3, growthRate=16,layer=3))
        self.middle.add_module("transition_3",transitionLayer(64,0.25))
        self.middle.add_module("upsampling_m", nn.ConvTranspose2d(16,16,2,2))
        
        self.up = nn.Sequential()
        self.up.add_module("bn_u1",nn.BatchNorm2d(32))
        self.up.add_module("conv_s1", nn.Conv2d(32, 16, 3, padding= 1))
        self.up.add_module("relu_s1", self.activ)
        self.up.add_module("conv_u1", denseBlock_origin(16, 3, growthRate=16, layer=3, bottleneckMulti=1,dilationLayer = dilate,activ = activation))
        self.up.add_module("transition_3",transitionLayer(64,0.25))
        self.up.add_module("upsampling_u", nn.ConvTranspose2d(16,16,2,2))
        
        self.block2 = nn.Sequential()
        self.block2.add_module("bn_3",nn.BatchNorm2d(32))
        self.block2.add_module("conv_s2", nn.Conv2d(32, 16, 3, padding= 1))
        self.block2.add_module("relu_s2", self.activ)
        self.block2.add_module("conv_3", denseBlock_origin(16, 3, growthRate=16, layer=3, bottleneckMulti=1,dilationLayer = dilate,activ = activation))
        self.block2.add_module("transition_3",transitionLayer(64,0.25))
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
            result = x8+x0[:,0:1,:,:]
        else:
            result = x8+x0
        
        return result