import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .networkUtil import *

# class DDU_withDC(nn.Module):
#     def __init__(self, inChannel = 3, activ = 'ReLU', fNum = 16):
#         super(DDU_withDC, self).__init__()
#         self.DDU = DilatedDenseUnet(inChannel, activ, fNum)
#         self.DC = dataConsistencyLayer()
    
#     def forward(self, x0, mask):
#         x1 = self.DDU(x0)
#         x2 = self.DC(x1, x0, mask)
        
#         return x2, x1

class DDU(nn.Module):
    def __init__(self, inChannel = 3, activ = 'ReLU', fNum=16):
        super(DDU, self).__init__()
        self.DDU = DilatedDenseUnet(inChannel, activ, fNum)

    def forward(self,x0):
        x1 = self.DDU(x0)

        return x1

class DDU_cn(nn.Module):
    def __init__(self, inChannel = 3, activ = 'ReLU', fNum=16, c = 2, addon = False, isComplex = False):
        super(DDU_cn, self).__init__()
        self.addon = addon
        templayerList = []
        for i in range(c):
            tmpDDU = DilatedDenseUnet(inChannel, activ, fNum, isComplex)
            tmpDF = dataConsistencyLayer(isStatic = True)
            templayerList.append(tmpDDU)
            templayerList.append(tmpDF)
        self.layerList = nn.ModuleList(templayerList)
        if(self.addon):
            self.addonDDU = DilatedDenseUnet(inChannel, activ, fNum, isComplex)

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

        if(self.addon):
            xt2 = xt
            xt = self.addonDDU(xt2)

        return xt,xt2

class DDU_c2(nn.Module):
    def __init__(self, inChannel = 3, activ = 'ReLU', fNum=16, isComplex = False):
        super(DDU_c2, self).__init__()
        self.DDU1 = DilatedDenseUnet(inChannel, activ, fNum, isComplex)
        self.DC1 = dataConsistencyLayer(isStatic = True)
        self.DDU2 = DilatedDenseUnet(inChannel, activ, fNum, isComplex)
        self.DC2 = dataConsistencyLayer(isStatic = True)

    def forward(self,x0,y,mask):
        x1 = self.DDU1(x0)
        x2 = self.DC1(x1,y,mask)
        x3 = self.DDU2(x2)
        x4 = self.DC2(x3,y,mask)

        return x4,x3

class DDU_debug(nn.Module):
    def __init__(self, inChannel = 3, activ = 'ReLU', fNum=16, isComplex = False):
        super(DDU_debug, self).__init__()
        self.DDU1 = DilatedDenseUnet_debug(inChannel, activ, fNum, isComplex)
        self.DC1 = dataConsistencyLayer(isStatic = True)
        self.DDU2 = DilatedDenseUnet_debug(inChannel, activ, fNum, isComplex)
        self.DC2 = dataConsistencyLayer(isStatic = True)

    def forward(self,x0,y,mask):
        x1 = self.DDU1(x0)
        x2 = self.DC1(x1,y,mask)
        x3 = self.DDU2(x2)
        x4 = self.DC2(x3,y,mask)

        return x4,x3

class DilatedDenseUnet(nn.Module):
    def __init__(self, inChannel = 3, activation = 'ReLU', fNumber = 16, isComplex = False):
        super(DilatedDenseUnet, self).__init__()
        dilate = True
        self.c = inChannel
        if(isComplex):
            chMulti = 2
        else:
            chMulti = 1
        
        if(activation == 'LeakyReLU'):
            self.activ = nn.LeakyReLU()
        elif(activation == 'ReLU'):
            self.activ = nn.ReLU()
        else:
            assert False,"wrong activation type"
        
        self.block0 = nn.Sequential()
        self.block0.add_module("conv_0", nn.Conv2d(inChannel*chMulti,fNumber,3,padding=1))
        self.block0.add_module("relu_0", self.activ)
        
        self.block1 = nn.Sequential()
        self.block1.add_module("conv_1", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        self.block1.add_module("conv_2", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        
        self.down = nn.Sequential()
        self.down.add_module("maxpool_d", nn.MaxPool2d(kernel_size=2))
        self.down.add_module("conv_d1", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        self.down.add_module("conv_d2", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))

        self.middle = nn.Sequential()
        self.middle.add_module("maxpool_m", nn.MaxPool2d(kernel_size=2))
        self.middle.add_module("conv_m1", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        self.middle.add_module("conv_m2", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        self.middle.add_module("upsampling_m", nn.ConvTranspose2d(fNumber,fNumber,2,2))
        
        self.up = nn.Sequential()
        self.up.add_module("bn_u1",nn.BatchNorm2d(fNumber*2))
        self.up.add_module("conv_s1", nn.Conv2d(fNumber*2, fNumber, 1, padding= 0))
        self.up.add_module("relu_s1", self.activ)
        self.up.add_module("conv_u1", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        self.up.add_module("conv_u2", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        self.up.add_module("upsampling_u", nn.ConvTranspose2d(fNumber,fNumber,2,2))
        
        self.block2 = nn.Sequential()
        self.block2.add_module("bn_3",nn.BatchNorm2d(fNumber*2))
        self.block2.add_module("conv_s2", nn.Conv2d(fNumber*2, fNumber, 1, padding= 0))
        self.block2.add_module("relu_s2", self.activ)
        self.block2.add_module("conv_3", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        self.block2.add_module("conv_4", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        self.block2.add_module("conv_5", nn.Conv2d(fNumber, 1*chMulti, 3, padding=1))
        
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


class DilatedDenseUnet_debug(nn.Module):
    def __init__(self, inChannel = 3, activation = 'ReLU', fNumber = 16, isComplex = False):
        super(DilatedDenseUnet_debug, self).__init__()
        dilate = True
        self.c = inChannel
        if(isComplex):
            chMulti = 2
        else:
            chMulti = 1
        
        if(activation == 'LeakyReLU'):
            self.activ = nn.LeakyReLU()
        elif(activation == 'ReLU'):
            self.activ = nn.ReLU()
        else:
            assert False,"wrong activation type"
        
        self.block0 = nn.Sequential()
        self.block0.add_module("conv_0", nn.Conv2d(inChannel*chMulti,fNumber,3,padding=1))
        self.block0.add_module("relu_0", self.activ)
        
        self.block1 = nn.Sequential()
        self.block1.add_module("conv_1", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        #self.block1.add_module("conv_2", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        
        self.down = nn.Sequential()
        self.down.add_module("maxpool_d", nn.MaxPool2d(kernel_size=2))
        self.down.add_module("conv_d1", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        #self.down.add_module("conv_d2", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))

        self.middle = nn.Sequential()
        self.middle.add_module("maxpool_m", nn.MaxPool2d(kernel_size=2))
        self.middle.add_module("conv_m1", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        #self.middle.add_module("conv_m2", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        self.middle.add_module("upsampling_m", nn.ConvTranspose2d(fNumber,fNumber,2,2))
        
        self.up = nn.Sequential()
        self.up.add_module("bn_u1",nn.BatchNorm2d(fNumber*2))
        self.up.add_module("conv_s1", nn.Conv2d(fNumber*2, fNumber, 1, padding= 0))
        self.up.add_module("relu_s1", self.activ)
        self.up.add_module("conv_u1", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        #self.up.add_module("conv_u2", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        self.up.add_module("upsampling_u", nn.ConvTranspose2d(fNumber,fNumber,2,2))
        
        self.block2 = nn.Sequential()
        self.block2.add_module("bn_3",nn.BatchNorm2d(fNumber*2))
        self.block2.add_module("conv_s2", nn.Conv2d(fNumber*2, fNumber, 1, padding= 0))
        self.block2.add_module("relu_s2", self.activ)
        self.block2.add_module("conv_3", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        #self.block2.add_module("conv_4", denseBlock(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        self.block2.add_module("conv_5", nn.Conv2d(fNumber, 1*chMulti, 3, padding=1))
        
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