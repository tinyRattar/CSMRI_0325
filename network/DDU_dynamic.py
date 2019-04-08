import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .networkUtil import * 

class DDU3d(nn.Module):
    def __init__(self, activ = 'ReLU', fNum=16):
        super(DDU3d, self).__init__()
        self.DDU3d = DilatedDenseUnet3d(activ, fNum)

    def forward(self,x0):
        x1 = self.DDU3d(x0)

        return x1

# class dataConsistencyLayer3d(nn.Module):
#     def __init__(self, initLamda = 1):
#         super(dataConsistencyLayer3d, self).__init__()
#         self.lamda = Parameter(torch.Tensor(1))
#         self.lamda.data.uniform_(0, 1)
#         #self.lamda.data = torch.Tensor([initLamda])
#         #self.lamda = initLamda
    
#     def forward(self, xin, y, mask):
#         iScale = self.lamda/(1+self.lamda)
#         mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],mask.shape[3],1)
#         if(xin.shape[1]==1):
#             emptyImag = torch.zeros_like(xin)
#             xin_c = torch.cat([xin,emptyImag],1).permute(0,2,3,4,1)
#             xGT_c = torch.cat([y,emptyImag],1).permute(0,2,3,4,1)
#         else:
#             xin_c = xin.permute(0,2,3,4,1)
#             xGT_c = y.permute(0,2,3,4,1)
        
#         xin_f = torch.fft(xin_c,2)
#         xGT_f = torch.fft(xGT_c,2)
        
#         xout_f = xin_f + (- xin_f + xGT_f) * iScale * mask

#         xout = torch.ifft(xout_f,2)
#         xout = xout.permute(0,4,1,2,3)
#         if(xin.shape[1]==1):
#             xout = xout[:,0:1]
        
#         return xout

class denseBlockLayer3d(nn.Module):
    def __init__(self,inChannel=64, outChannel=64, kernelSize=3, inception = False, dilateScale = 1, activ = 'ReLU'):
        super(denseBlockLayer3d, self).__init__()
        self.useInception = inception

        if(self.useInception):
            self.conv1 = nn.Conv3d(inChannel,outChannel,3,padding = 1)
            self.conv2 = nn.Conv3d(inChannel,outChannel,5,padding = 2)
            self.conv3 = nn.Conv3d(inChannel,outChannel,7,padding = 3)
            if(activ == 'LeakyReLU'):
                self.relu = nn.LeakyReLU()
            else:
                self.relu = nn.ReLU()
            self.conv4 = nn.Conv3d(outChannel*3,outChannel,1,padding = 0)
            #self.relu2 = nn.ReLU()
        else:
            pad = int(dilateScale * (kernelSize - 1) / 2)
            
            self.conv = nn.Conv3d(inChannel,outChannel,kernelSize,padding = pad, dilation = dilateScale)
            if(activ == 'LeakyReLU'):
                self.relu = nn.LeakyReLU()
            else:
                self.relu = nn.ReLU()
            
    def forward(self,x):
        if(self.useInception):
            y2 = x
            y3_1 = self.conv1(y2)
            y3_2 = self.conv1(y2)
            y3_3 = self.conv1(y2)
            y4 = torch.cat((y3_1,y3_2,y3_3),1)
            y4 = self.relu(y4)
            y5 = self.conv4(y4)
            y_ = self.relu(y5)
        else:
            y2 = self.conv(x)
            y_ = self.relu(y2)
            
            
        return y_
    
class denseBlock3d(nn.Module):
    def __init__(self, inChannel=64, outChannel=64, kernelSize=3, growthRate=16, layer=4, inceptionLayer = False, dilationLayer = False, activ = 'ReLU'):
        super(denseBlock3d, self).__init__()
        dilate = 1
        if(dilationLayer):
            dilateMulti = 2
        else:
            dilateMulti = 1
        pad = int((kernelSize-1)/2)
        self.layer = layer
        templayerList = []
        for i in range(0, layer):
            tempLayer = denseBlockLayer3d(inChannel+growthRate*i,growthRate,kernelSize,inceptionLayer,dilate,activ)
            dilate = dilate * dilateMulti
            templayerList.append(tempLayer)
        self.layerList = nn.ModuleList(templayerList)
        self.outputLayer = denseBlockLayer3d(inChannel+growthRate*layer,outChannel,kernelSize,inceptionLayer)
        self.bn = nn.BatchNorm3d(outChannel)
            
    def forward(self,x):
        for i in range(0, self.layer):
            tempY = self.layerList[i](x)
            x = torch.cat((x, tempY), 1)
        y = self.outputLayer(x)
        y = self.bn(y)
            
        return y
    
class DilatedDenseUnet3d(nn.Module):
    def __init__(self, activation = 'ReLU', fNumber = 16):
        super(DilatedDenseUnet3d, self).__init__()
        dilate = True
        
        if(activation == 'LeakyReLU'):
            self.activ = nn.LeakyReLU()
        elif(activation == 'ReLU'):
            self.activ = nn.ReLU()
        else:
            assert False,"wrong activation type"
        
        self.block0 = nn.Sequential()
        self.block0.add_module("conv_0", nn.Conv3d(1,fNumber,3,padding=1))
        self.block0.add_module("relu_0", self.activ)
        
        self.block1 = nn.Sequential()
        self.block1.add_module("conv_1", denseBlock3d(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        self.block1.add_module("conv_2", denseBlock3d(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        
        self.down = nn.Sequential()
        self.down.add_module("maxpool_d", nn.MaxPool3d((1, 2, 2)))
        self.down.add_module("conv_d1", denseBlock3d(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        self.down.add_module("conv_d2", denseBlock3d(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))

        self.middle = nn.Sequential()
        self.middle.add_module("maxpool_m", nn.MaxPool3d((1, 2, 2)))
        self.middle.add_module("conv_m1", denseBlock3d(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        self.middle.add_module("conv_m2", denseBlock3d(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        self.middle.add_module("upsampling_m", nn.ConvTranspose3d(fNumber,fNumber,(1,2,2),(1,2,2)))
        
        self.up = nn.Sequential()
        self.up.add_module("bn_u1",nn.BatchNorm3d(fNumber*2))
        self.up.add_module("conv_s1", nn.Conv3d(fNumber*2, fNumber, 1, padding= 0))
        self.up.add_module("relu_s1", self.activ)
        self.up.add_module("conv_u1", denseBlock3d(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        self.up.add_module("conv_u2", denseBlock3d(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        self.up.add_module("upsampling_u", nn.ConvTranspose3d(fNumber,fNumber,(1,2,2),(1,2,2)))
        
        self.block2 = nn.Sequential()
        self.block2.add_module("bn_3",nn.BatchNorm3d(fNumber*2))
        self.block2.add_module("conv_s2", nn.Conv3d(fNumber*2, fNumber, 1, padding= 0))
        self.block2.add_module("relu_s2", self.activ)
        self.block2.add_module("conv_3", denseBlock3d(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        self.block2.add_module("conv_4", denseBlock3d(fNumber, fNumber, 3, growthRate=16,layer=3,dilationLayer = dilate,activ = activation))
        self.block2.add_module("conv_5", nn.Conv3d(fNumber, 1, 3, padding = 1))
        
    def forward(self,x0):
        x1 = self.block0(x0)
        x2 = self.block1(x1)
        x3 = self.down(x2)
        x4 = self.middle(x3)
        
        x5 = torch.cat((x3,x4),1)
        x6 = self.up(x5)
        
        x7 = torch.cat((x2,x6),1)
        x8 = self.block2(x7)
        
        result = x8+x0
        
        return result
