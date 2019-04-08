import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .networkUtil import *

#todo: dilation is not in such usage
class DenseUnet_dilate(nn.Module):
    def __init__(self, inChannel = 3):
        super(DenseUnet_dilate, self).__init__()
        self.c = inChannel
        
        self.block0 = nn.Sequential()
        self.block0.add_module("conv_0", nn.Conv2d(inChannel,16,3,padding=1))
        
        self.block1 = nn.Sequential()
        ## WARNING: bottleneckMulti has decreased for debug
        self.block1.add_module("conv_1", denseBlock_dilate(16, 3, growthRate=16, layer=3, bottleneckMulti=1))
        self.block1.add_module("transition_1",transitionLayer(64,0.25))
        
        self.down = nn.Sequential()
        self.down.add_module("maxpool_d", nn.MaxPool2d(kernel_size=2))
        self.down.add_module("conv_d1", denseBlock_dilate(16, 3, growthRate=16, layer=3, bottleneckMulti=1))
        self.down.add_module("transition_2",transitionLayer(64,0.25))

        self.middle = nn.Sequential()
        self.middle.add_module("maxpool_m", nn.MaxPool2d(kernel_size=2))
        self.middle.add_module("conv_m1", denseBlock_dilate(16, 3, growthRate=16,layer=3))
        self.middle.add_module("transition_3",transitionLayer(64,0.25))
        self.middle.add_module("upsampling_m", nn.ConvTranspose2d(16,16,2,2))
        
        self.up = nn.Sequential()
        self.up.add_module("bn_u1",nn.BatchNorm2d(32))
        self.up.add_module("conv_s1", nn.Conv2d(32, 16, 3, padding= 1))
        self.up.add_module("relu_s1", nn.ReLU())
        self.up.add_module("conv_u1", denseBlock_dilate(16, 3, growthRate=16, layer=3, bottleneckMulti=1))
        self.up.add_module("transition_3",transitionLayer(64,0.25))
        self.up.add_module("upsampling_u", nn.ConvTranspose2d(16,16,2,2))
        
        self.block2 = nn.Sequential()
        self.block2.add_module("bn_3",nn.BatchNorm2d(32))
        self.block2.add_module("conv_s2", nn.Conv2d(32, 16, 3, padding= 1))
        self.block2.add_module("relu_s2", nn.ReLU())
        self.block2.add_module("conv_3", denseBlock_dilate(16, 3, growthRate=16, layer=3, bottleneckMulti=1))
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
            result = x8+x1[:,1:2,:,:]
        else:
            result = x8+x1
        
        return result