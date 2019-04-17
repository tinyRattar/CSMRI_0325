import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .networkUtil import *
from .networkUtil_dynamic import *

class convBlock3d(nn.Module):
    def __init__(self, inChannel = 2, iConvNum = 5, f=32, dilationLayer = False):
        super(convBlock3d, self).__init__()
        dilate = 1
        if(dilationLayer):
            dilateMulti = 2
        else:
            dilateMulti = 1
        self.Relu = nn.ReLU()
        self.conv1 = nn.Conv3d(inChannel,f,3,padding = 1)
        convList = []
        for i in range(1, iConvNum-1):
            tmpConv = nn.Conv3d(f,f,3,padding = dilate, dilation = dilate)
            dilate = dilate * dilateMulti
            convList.append(tmpConv)
        self.layerList = nn.ModuleList(convList)
        self.conv2 = nn.Conv3d(f,2,3,padding = 1)
    
    def forward(self, x1):
        x2 = self.conv1(x1)
        x2 = self.Relu(x2)
        for conv in self.layerList:
            x2 = conv(x2)
            x2 = self.Relu(x2)
        x3 = self.conv2(x2)
        #x3 = self.Relu(x3)
        
        x4 = x3 + x1[:,0:2]
        
        return x4

class denseConv3d(nn.Module):
    def __init__(self, inChannel=16, outChannel=16, kernelSize=3, growthRate=16, layer=4, inceptionLayer = False, dilationLayer = False, activ = 'ReLU', useOri = False):
        super(denseConv3d, self).__init__()
        dilate = 1
        if(dilationLayer):
            dilateMulti = 2
        else:
            dilateMulti = 1
        pad = int((kernelSize-1)/2)
        self.denselayer = layer-2
        templayerList = []
        for i in range(0, self.denselayer):
            if(useOri):
                tempLayer = denseBlockLayer_origin3d(inChannel+growthRate*i, growthRate, kernelSize, inChannel, dilate, activ)
            else:
                tempLayer = denseBlockLayer3d(inChannel+growthRate*i,growthRate,kernelSize,inceptionLayer,dilate,activ)
            dilate = dilate * dilateMulti
            templayerList.append(tempLayer)
        self.layerList = nn.ModuleList(templayerList)
            
    def forward(self,x):
        for i in range(0, self.denselayer):
            tempY = self.layerList[i](x)
            x = torch.cat((x, tempY), 1)
            
        return x

class subDenseNet3d(nn.Module):
    '''
    ioChannel[0] = 2 for complex
    '''
    def __init__(self, ioChannel = 1, fNum = 16, growthRate = 16, layer = 3, dilate = False, activation = 'ReLU', useOri = False, transition = 0):
        super(subDenseNet3d, self).__init__()
        if(isinstance(ioChannel,int)):
        	self.inChannel = ioChannel
        	self.outChannel = ioChannel
        else:
        	self.inChannel = ioChannel[0]
        	self.outChannel = ioChannel[1]

        self.transition = transition
        self.inConv = nn.Conv3d(self.inChannel, fNum, 3, padding = 1)
        if(activation == 'LeakyReLU'):
            self.activ = nn.LeakyReLU()
        elif(activation == 'ReLU'):
            self.activ = nn.ReLU()
        self.denseConv = denseConv3d(fNum, fNum, 3, growthRate, layer, dilationLayer = dilate, activ = activation, useOri = useOri)
        if(transition>0):
            #assert transition==0.5, "transition has to be 0.5 for debug"
            self.transitionLayer = transitionLayer3d(fNum+growthRate*(layer-2), transition, activ = activation)
            self.outConv = convLayer3d(int((fNum+growthRate*(layer-2))*transition), self.outChannel, activ = activation)
        else:
            self.outConv = convLayer3d(fNum+growthRate*(layer-2), self.outChannel, activ = activation)

    def forward(self,x):
        x2 = self.inConv(x)
        #x2 = self.activ(x2)
        x2 = self.denseConv(x2)
        if(self.transition>0):
        	x2 = self.transitionLayer(x2)
        	#x2 = self.activ(x2)
        x2 = self.outConv(x2)
        x3 = x2+x[:,:self.outChannel]

        return x3

class CN_Dense_3d(nn.Module):
    def __init__(self, inChannel = 1, d = 5, c = 5, fNum = 16, growthRate = 16, dilate = False, activation = 'ReLU', useOri = False, transition=0, trick = 0, globalDense = False):
        super(CN_Dense_3d, self).__init__()
        self.globalSkip = globalDense
        templayerList = []
        for i in range(c):
            if(self.globalSkip):
                tmpSubNet = subDenseNet3d((inChannel*(i+1),inChannel), fNum, growthRate, d, dilate, activation, useOri, transition)
            else:
                tmpSubNet = subDenseNet3d(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition)
            tmpDF = dataConsistencyLayer_static(trick = trick, dynamic = True)
            templayerList.append(tmpSubNet)
            templayerList.append(tmpDF)
        self.layerList = nn.ModuleList(templayerList)
        
    def forward(self, x1, y, mask):
        xin = x1
        flag = True
        for layer in self.layerList:
            if(flag):
                xt = layer(xin)
                flag = False
            else:
                xt = layer(xt, y, mask)
                flag = True
                if(self.globalSkip):
                	xin = torch.cat([xt,xin],1)
                else:
                	xin = xt
        
        return xt

class CN_Conv_3d(nn.Module):
    def __init__(self, inChannel = 2, d = 5, c = 5, fNum = 32, dilate = False, trick = 0):
        '''
        trick 1 : abs->DC
        trick 2 : DC->abs->DC
        '''
        self.trick = trick
        super(CN_Conv_3d, self).__init__()
        templayerList = []
        for i in range(c):
            tmpSubNet = convBlock3d(2, d, fNum, dilate)
            tmpDF = dataConsistencyLayer_static(dynamic = True)
            templayerList.append(tmpSubNet)
            templayerList.append(tmpDF)
        self.layerList = nn.ModuleList(templayerList)

        if(self.trick in [3,4]):
            tmpConvList = []
            for i in range(1):
                tempConv = nn.Conv3d(4,2,1,padding=0)
                tmpConvList.append(tempConv)
            self.trickConvList = nn.ModuleList(tmpConvList)
        
    def forward(self, x1, y, mask):
        xt = x1
        flag = True
        index = 0
        for layer in self.layerList:
            if(flag):
                xt = layer(xt)
                flag = False
            else:
                if(self.trick == 1):
                    xt = abs4complex(xt)
                    xt = layer(xt, y, mask)
                elif(self.trick == 2):
                    xt = layer(xt, y, mask)
                    xt = abs4complex(xt)
                    xt = layer(xt, y, mask)
                elif(self.trick == 3):
                    xdc1 = layer(xt, y, mask)
                    xt = abs4complex(xt)
                    xdc2 = layer(xt, y, mask)
                    xdc = torch.cat([xdc1,xdc2],1)
                    xt = self.trickConvList[0](xdc)
                    #index += 1
                elif(self.trick == 4):
                    xdc1 = layer(xt, y, mask)
                    xabs = abs4complex(xdc1)
                    xdc2 = layer(xabs, y, mask)
                    xdc = torch.cat([xdc1,xdc2],1)
                    xt = self.trickConvList[0](xdc)
                    #index += 1
                else:
                    xt = layer(xt, y, mask)
                flag = True
        
        return xt

class CN_SkipConv_3d(nn.Module):
    def __init__(self, inChannel = 2, d = 5, c = 5, fNum = 32, dilate = False, trick = 0, skipMode = 0):
        super(CN_SkipConv_3d, self).__init__()
        assert c==5, "only c = 5 is acceptable in skip mode"
        self.trick = trick
        self.skipMode = skipMode
        
        templayerList = []
        for i in range(c):
            if(i<2 or (i<3 and self.skipMode==0)):
                tmpSubNet = convBlock3d(2, d, fNum, dilate)
            else:
                tmpSubNet = convBlock3d(4, d, fNum, dilate)
            tmpDF = dataConsistencyLayer_static(dynamic = True)
            templayerList.append(tmpSubNet)
            templayerList.append(tmpDF)
        self.layerList = nn.ModuleList(templayerList)

        if(self.trick in [3,4]):
            tmpConvList = []
            for i in range(c):
                tempConv = nn.Conv3d(4,2,1,padding=0)
                tmpConvList.append(tempConv)
            self.trickConvList = nn.ModuleList(tmpConvList)

    def trickForward(self,x,y,mask,layer,index):
        xt = x
        if(self.trick == 1):
            xt = abs4complex(xt)
            xt = layer(xt, y, mask)
        elif(self.trick == 2):
            xt = layer(xt, y, mask)
            xt = abs4complex(xt)
            xt = layer(xt, y, mask)
        elif(self.trick == 3):
            xdc1 = layer(xt, y, mask)
            xt = abs4complex(xt)
            xdc2 = layer(xt, y, mask)
            xdc = torch.cat([xdc1,xdc2],1)
            xt = self.trickConvList[index](xdc)
        elif(self.trick == 4):
            xdc1 = layer(xt, y, mask)
            xabs = abs4complex(xdc1)
            xdc2 = layer(xabs, y, mask)
            xdc = torch.cat([xdc1,xdc2],1)
            xt = self.trickConvList[index](xdc)
        else:
            xt = layer(xt, y, mask)

        return xt
        
    def forward(self, x1, y, mask):
        if(self.skipMode == 0):
            x2 = self.layerList[0](x1)
            x3 = self.trickForward(x2,y,mask,self.layerList[1],0)

            x4 = self.layerList[2](x3)
            x5 = self.trickForward(x4,y,mask,self.layerList[3],1)

            x6 = self.layerList[4](x5)
            x7 = self.trickForward(x6,y,mask,self.layerList[5],2)

            x7c = torch.cat([x7,x5],1)
            x8 = self.layerList[6](x7c)
            x9 = self.trickForward(x8,y,mask,self.layerList[7],3)

            x9c = torch.cat([x9,x3],1)
            x10 = self.layerList[8](x9c)
            x11 = self.trickForward(x10,y,mask,self.layerList[9],4)

            return x11
        else:
            x2 = self.layerList[0](x1)
            x3 = self.trickForward(x2,y,mask,self.layerList[1],0)

            x4 = self.layerList[2](x3)
            x5 = self.trickForward(x4,y,mask,self.layerList[3],1)

            x5c = torch.cat([x5,x4],1)
            x6 = self.layerList[4](x5c)
            x7 = self.trickForward(x6,y,mask,self.layerList[5],2)

            x7c = torch.cat([x7,x2],1)
            x8 = self.layerList[6](x7c)
            x9 = self.trickForward(x8,y,mask,self.layerList[7],3)

            x9c = torch.cat([x9,x1],1)
            x10 = self.layerList[8](x9c)
            x11 = self.trickForward(x10,y,mask,self.layerList[9],4)

            return x11