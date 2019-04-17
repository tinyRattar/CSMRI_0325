import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

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
    
# class SELayer(nn.Module):
#     def __init__(self,channel,height,width):
#         super(SELayer, self).__init__()
#         self.inShape = [channel,height,width]
#         self.globalPooling = nn.AvgPool2d((self.inShape[1],self.inShape[2]))
#         self.fc1 = nn.Conv2d(self.inShape[0],int(self.inShape[0]/2),1)
#         self.fc2 = nn.Conv2d(int(self.inShape[0]/2),self.inShape[0],1)
    
#     def forward(self,x):
#         v1 = self.globalPooling(x)
#         v2 = self.fc1(v1)
#         v3 = self.fc2(v2)
        
#         f = x*v3
        
#         return f
    
class denseBlockLayer_origin3d(nn.Module):
    def __init__(self,inChannel=64, outChannel=64, kernelSize = 3, bottleneckChannel = 64, dilateScale = 1, activ = 'ReLU'):
        super(denseBlockLayer_origin3d, self).__init__()
        pad = int((kernelSize-1)/2)

        self.bn = nn.BatchNorm3d(inChannel)
        if(activ == 'LeakyReLU'):
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()
        self.conv = nn.Conv3d(inChannel,bottleneckChannel,1)
        
        self.bn2 = nn.BatchNorm3d(bottleneckChannel)
        if(activ == 'LeakyReLU'):
            self.relu2 = nn.LeakyReLU()
        else:
            self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv3d(bottleneckChannel,outChannel,kernelSize,padding = dilateScale,dilation=dilateScale)
        
            
    def forward(self,x):
        x1 = self.bn(x)
        x1 = self.relu(x1)
        x1 = self.conv(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        y = self.conv2(x1)
            
        return y
    
class transitionLayer3d(nn.Module):
    def __init__(self, inChannel = 64, compressionRate = 0.5, activ = 'ReLU'):
        super(transitionLayer3d, self).__init__()
        self.bn = nn.BatchNorm3d(inChannel)
        if(activ == 'LeakyReLU'):
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()
        self.conv = nn.Conv3d(inChannel,int(inChannel*compressionRate),1)
        
    def forward(self,x):
        x1 = self.bn(x)
        x2 = self.relu(x1)
        y = self.conv(x2)
        
        return y

class convLayer3d(nn.Module):
    def __init__(self, inChannel = 64, outChannel = 64, activ = 'ReLU'):
        super(convLayer3d, self).__init__()
        self.bn = nn.BatchNorm3d(inChannel)
        if(activ == 'LeakyReLU'):
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()
        self.conv = nn.Conv3d(inChannel,outChannel,3,padding = 1)
        
    def forward(self,x):
        x1 = self.bn(x)
        x2 = self.relu(x1)
        y = self.conv(x2)
        
        return y