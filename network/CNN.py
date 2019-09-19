import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .networkUtil import *

class vanillaCNN(nn.Module):
    def __init__(self):
        super(vanillaCNN,self).__init__()
        self.convBlock = fourConv()

    def forward(self,x1):
        x2 = self.convBlock(x1)
        x2 = x2+x1

        return x2

class fourConv(nn.Module):
    def __init__(self, dilate=False):
        super(fourConv, self).__init__()
        self.relu = nn.ReLU()
        if(dilate):
            self.conv1 = nn.Conv2d(16,32,3,padding = 1)
            self.conv2 = nn.Conv2d(32,64,3,padding = 2, dilation=2)
            self.conv3 = nn.Conv2d(64,32,3,padding = 4, dilation=4)
            self.conv4 = nn.Conv2d(32,16,3,padding = 8, dilation=8)
        else:
            self.conv1 = nn.Conv2d(1,32,3,padding = 1)
            self.conv2 = nn.Conv2d(32,64,3,padding = 1)
            self.conv3 = nn.Conv2d(64,32,3,padding = 1)
            self.conv4 = nn.Conv2d(32,1,3,padding = 1)
    
    
    def forward(self,x1):
        x2 = self.conv1(x1)
        x2 = self.relu(x2)
        x3 = self.conv2(x2)
        x3 = self.relu(x3)
        x4 = self.conv3(x3)
        x4 = self.relu(x4)
        x5 = self.conv4(x4)
        #x5 = x5+x1
        
        return x5
        

class CNN(nn.Module):
    def __init__(self, dilationLayer = False):
        super(CNN, self).__init__()
        
        self.block0 = nn.Sequential()
        self.block0.add_module("conv_0", nn.Conv2d(3,16,3,padding = 1))
        self.block0.add_module("relu_0", nn.ReLU())
        
        self.block1 = nn.Sequential()
        self.block1.add_module("conv_1", fourConv(dilationLayer))
        
        self.down = nn.Sequential()
        self.down.add_module("maxpool_d", nn.MaxPool2d(kernel_size=2))
        self.down.add_module("conv_d1", fourConv(dilationLayer))

        self.middle = nn.Sequential()
        self.middle.add_module("maxpool_m", nn.MaxPool2d(kernel_size=2))
        self.middle.add_module("conv_m1", fourConv(dilationLayer))
        self.middle.add_module("upsampling_m", nn.ConvTranspose2d(16,16,2,2))
        
        self.up = nn.Sequential()
        self.up.add_module("bn_u1",nn.BatchNorm2d(32))
        self.up.add_module("conv_s1", nn.Conv2d(32, 16, 1, padding= 0))
        self.up.add_module("relu_s1", nn.ReLU())
        self.up.add_module("conv_u1", fourConv(dilationLayer))
        self.up.add_module("upsampling_u", nn.ConvTranspose2d(16,16,2,2))
        
        self.block2 = nn.Sequential()
        self.block2.add_module("bn_3",nn.BatchNorm2d(32))
        self.block2.add_module("conv_s2", nn.Conv2d(32, 16, 1, padding= 0))
        self.block2.add_module("relu_s2", nn.ReLU())
        self.block2.add_module("conv_3", fourConv(dilationLayer))
        self.block2.add_module("conv_5", nn.Conv2d(16, 1, 1, padding=0))

        self.DC = dataConsistencyLayer(isStatic = True)
        
    def forward(self,x1,y,mask):
        x1 = self.block0(x1)
        x2 = self.block1(x1)
        x3 = self.down(x2)
        x4 = self.middle(x3)
        
        x5 = torch.cat((x3,x4),1)
        x6 = self.up(x5)
        
        x7 = torch.cat((x2,x6),1)
        x8 = self.block2(x7)
        
        result = x8+x1[:,1:2,:,:]
        result = self.DC(result,y,mask)
        
        return result

class Unet_dc(nn.Module):
    def __init__(self, dilationLayer = False):
        super(Unet_dc, self).__init__()
        
        self.block0 = nn.Sequential()
        self.block0.add_module("conv_0", nn.Conv2d(2,64,3,padding = 1))
        self.block0.add_module("relu_0", nn.ReLU())
        
        self.block1 = nn.Sequential()
        self.block1.add_module("conv_1", nn.Conv2d(64,64,3,padding = 1))
        self.block1.add_module("relu_1", nn.ReLU())
        
        self.down = nn.Sequential()
        self.down.add_module("maxpool_d", nn.MaxPool2d(kernel_size=2))
        self.down.add_module("conv_d1", nn.Conv2d(64,128,3,padding = 1))
        self.down.add_module("relu_d1", nn.ReLU())
        self.down.add_module("conv_d2", nn.Conv2d(128,128,3,padding = 1))
        self.down.add_module("relu_d2", nn.ReLU())

        self.middle = nn.Sequential()
        self.middle.add_module("maxpool_m", nn.MaxPool2d(kernel_size=2))
        self.middle.add_module("conv_m1", nn.Conv2d(128,256,3,padding = 1))
        self.middle.add_module("relu_m1", nn.ReLU())
        self.middle.add_module("conv_m2", nn.Conv2d(256,256,3,padding = 1))
        self.middle.add_module("relu_m2", nn.ReLU())
        self.middle.add_module("conv_m3", nn.Conv2d(256,128,3,padding = 1))
        self.middle.add_module("relu_m3", nn.ReLU())
        self.middle.add_module("upsampling_m", nn.ConvTranspose2d(128,128,2,2))
        
        self.up = nn.Sequential()
        self.up.add_module("conv_s1", nn.Conv2d(256, 128, 1, padding= 0))
        self.up.add_module("relu_s1", nn.ReLU())
        self.up.add_module("conv_s2", nn.Conv2d(128, 64, 1, padding= 0))
        self.up.add_module("relu_s2", nn.ReLU())
        self.up.add_module("upsampling_u", nn.ConvTranspose2d(64,64,2,2))
        
        self.block2 = nn.Sequential()
        self.block2.add_module("conv_s3", nn.Conv2d(128, 64, 1, padding= 0))
        self.block2.add_module("relu_s3", nn.ReLU())
        self.block2.add_module("conv_s4", nn.Conv2d(64, 64, 1, padding= 0))
        self.block2.add_module("relu_s4", nn.ReLU())
        self.block2.add_module("conv_5", nn.Conv2d(64, 2, 1, padding=0))
        
        self.DC = dataConsistencyLayer(isStatic = True)


    def forward(self,x0,y,mask):
        x1 = self.block0(x0)
        x2 = self.block1(x1)
        x3 = self.down(x2)
        x4 = self.middle(x3)
        
        x5 = torch.cat((x3,x4),1)
        x6 = self.up(x5)
        
        x7 = torch.cat((x2,x6),1)
        x8 = self.block2(x7)
        
        result = x8+x0

        result = self.DC(result,y,mask)
        
        return result