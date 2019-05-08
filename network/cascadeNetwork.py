import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .networkUtil import *
from util import gaussianFilter

class convBlock(nn.Module):
    def __init__(self, ioChannel = 2, iConvNum = 5, f=32, dilationLayer = False):
        super(convBlock, self).__init__()
        if(isinstance(ioChannel,int)):
            self.inChannel = ioChannel
            self.outChannel = ioChannel
        else:
            self.inChannel = ioChannel[0]
            self.outChannel = ioChannel[1]
        dilate = 1
        if(dilationLayer):
            dilateMulti = 2
        else:
            dilateMulti = 1
        self.Relu = nn.ReLU()
        self.conv1 = nn.Conv2d(self.inChannel,f,3,padding = 1)
        convList = []
        for i in range(1, iConvNum-1):
            tmpConv = nn.Conv2d(f,f,3,padding = dilate, dilation = dilate)
            dilate = dilate * dilateMulti
            convList.append(tmpConv)
        self.layerList = nn.ModuleList(convList)
        self.conv2 = nn.Conv2d(f,self.outChannel,3,padding = 1)
    
    def forward(self, x1):
        x2 = self.conv1(x1)
        x2 = self.Relu(x2)
        for conv in self.layerList:
            x2 = conv(x2)
            x2 = self.Relu(x2)
        x3 = self.conv2(x2)
        #x3 = self.Relu(x3)
        
        x4 = x3 + x1[:,0:self.outChannel]
        
        return x4

class denseConv(nn.Module):
    def __init__(self, inChannel=16, kernelSize=3, growthRate=16, layer=4, inceptionLayer = False, dilationLayer = False, activ = 'ReLU', useOri = False):
        super(denseConv, self).__init__()
        dilate = 1
        if(dilationLayer):
            dilateMulti = 2
        else:
            dilateMulti = 1
        pad = int((kernelSize-1)/2)
        #self.denselayer = layer-2
        self.denselayer = layer
        templayerList = []
        for i in range(0, self.denselayer):
            if(useOri):
                tempLayer = denseBlockLayer_origin(inChannel+growthRate*i, growthRate, kernelSize, inChannel, dilate, activ)
            else:
                tempLayer = denseBlockLayer(inChannel+growthRate*i, growthRate, kernelSize, inceptionLayer, dilate, activ)
            dilate = dilate * dilateMulti
            templayerList.append(tempLayer)
        self.layerList = nn.ModuleList(templayerList)
            
    def forward(self,x):
        for i in range(0, self.denselayer):
            tempY = self.layerList[i](x)
            x = torch.cat((x, tempY), 1)
            
        return x

class subDenseNet(nn.Module):
    '''
    ioChannel[0] = 2 for complex
    '''
    def __init__(self, ioChannel = 2, fNum = 16, growthRate = 16, layer = 3, dilate = False, activation = 'ReLU', useOri = False, transition = 0, useSE = False, residual = True):
        super(subDenseNet, self).__init__()
        if(isinstance(ioChannel,int)):
            self.inChannel = ioChannel
            self.outChannel = ioChannel
        else:
            self.inChannel = ioChannel[0]
            self.outChannel = ioChannel[1]

        self.transition = transition
        self.useSE = useSE
        self.residual = residual
        self.inConv = nn.Conv2d(self.inChannel, fNum, 3, padding = 1)
        if(activation == 'LeakyReLU'):
            self.activ = nn.LeakyReLU()
        elif(activation == 'ReLU'):
            self.activ = nn.ReLU()
        self.denseConv = denseConv(fNum, 3, growthRate, layer - 2, dilationLayer = dilate, activ = activation, useOri = useOri)
        if(self.useSE):
        	self.se = SELayer(fNum+growthRate*(layer-2),256,256)
        if(transition>0):
            #assert transition==0.5, "transition has to be 0.5 for debug"
            self.transitionLayer = transitionLayer(fNum+growthRate*(layer-2), transition, activ = activation)
            #self.outConv = nn.Conv2d(int((fNum+growthRate*(layer-2))*transition), inChannel, 3, padding = 1)
            self.outConv = convLayer(int((fNum+growthRate*(layer-2))*transition), self.outChannel, activ = activation)
        else:
            #self.outConv = nn.Conv2d(fNum+growthRate*(layer-2), inChannel, 3, padding = 1)
            self.outConv = convLayer(fNum+growthRate*(layer-2), self.outChannel, activ = activation)

    def forward(self, x):
        x2 = self.inConv(x)
        #x2 = self.activ(x2)
        x2 = self.denseConv(x2)
        if(self.useSE):
        	x2 = self.se(x2)
        if(self.transition>0):
            x2 = self.transitionLayer(x2)
            #x2 = self.activ(x2)
        x2 = self.outConv(x2)
        if(self.residual):
            x2 = x2+x[:,:self.outChannel]

        return x2

class subUnet(nn.Module):
    def __init__(self, ioChannel = 2, fNum = 16, growthRate = 16, layer = 3, dilate = False, activation = 'ReLU', useOri = False, transition = 0.5, useSE = False, residual = True):
        super(subUnet, self).__init__()
        if(isinstance(ioChannel,int)):
            self.inChannel = ioChannel
            self.outChannel = ioChannel
        else:
            self.inChannel = ioChannel[0]
            self.outChannel = ioChannel[1]
        #self.transition = transition
        self.useSE = useSE
        self.residual = residual
        self.inConv = nn.Conv2d(self.inChannel, fNum, 3, padding = 1)

        self.e1_inter = denseConv(fNum, 3, growthRate, layer, dilationLayer = dilate, activ = activation, useOri = useOri)
        self.e1_tr = convLayer(fNum+growthRate*layer, fNum, activ = activation, kernelSize = 1)
        self.e1_ds = nn.MaxPool2d(kernel_size = 2)

        self.e2_inter = denseConv(fNum, 3, growthRate, layer, dilationLayer = dilate, activ = activation, useOri = useOri)
        self.e2_tr = convLayer(fNum+growthRate*layer, fNum, activ = activation, kernelSize = 1)
        self.e2_ds = nn.MaxPool2d(kernel_size = 2)

        self.m_inter = denseConv(fNum, 3, growthRate, layer, dilationLayer = dilate, activ = activation, useOri = useOri)
        self.m_tr = convLayer(fNum+growthRate*layer, fNum, activ = activation, kernelSize = 1)
        self.m_us = nn.ConvTranspose2d(fNum, fNum, 2, 2)

        self.d2_cat = convLayer(2 * fNum, fNum, activ = activation, kernelSize = 1)
        self.d2_inter = denseConv(fNum, 3, growthRate, layer, dilationLayer = dilate, activ = activation, useOri = useOri)
        self.d2_tr = convLayer(fNum+growthRate*layer, fNum, activ = activation, kernelSize = 1)
        self.d2_us = nn.ConvTranspose2d(fNum, fNum, 2, 2)

        self.d1_cat = convLayer(2 * fNum, fNum, activ = activation, kernelSize = 1)
        self.d1_inter = denseConv(fNum, 3, growthRate, layer, dilationLayer = dilate, activ = activation, useOri = useOri)
        self.d1_tr = convLayer(fNum+growthRate*layer, fNum, activ = activation, kernelSize = 1)

        #self.outConv = nn.Conv2d(fNum, self.outChannel, 3, padding = 1)
        self.outConv = convLayer(fNum, self.outChannel, activ = activation, kernelSize = 3)

    def forward(self, x):
        fin = self.inConv(x)
        f2 = self.e1_tr(self.e1_inter(fin))
        f2d = self.e1_ds(f2)

        f3 = self.e2_tr(self.e2_inter(f2d))
        f3d = self.e2_ds(f3)

        f4 = self.m_tr(self.m_inter(f3d))
        f4u = self.m_us(f4)

        f4c = torch.cat([f4u,f3],1)

        f5 = self.d2_cat(f4c)
        f5 = self.d2_tr(self.d2_inter(f5))
        f5u = self.d2_us(f5)

        f5c = torch.cat([f5u,f2],1)

        f6 = self.d1_cat(f5c)
        f6 = self.d1_tr(self.d1_inter(f6))

        y = self.outConv(f6)

        return y

class CN_Dense(nn.Module):
    def __init__(self, inChannel = 1, d = 5, c = 5, fNum = 16, growthRate = 16, dilate = False, activation = 'ReLU',  \
        useOri = False, transition = 0, trick = 0, globalDense = False, useSE = False, globalResSkip = False, subnetType = 'Dense'):
        super(CN_Dense, self).__init__()
        self.globalSkip = globalDense
        self.globalResSkip = globalResSkip
        if(isinstance(trick,int)):
            trickList = [trick] * c
        else:
            assert len(trick) == c, 'Different length of c and trick'
            trickList = trick
        if(isinstance(subnetType,str)):
            subnetType = [subnetType] * c
        else:
            assert len(subnetType) == c, 'Different length of c and subnetType'
        subNetClassList = []
        for netType in subnetType:
            if(netType == 'Dense' or netType == 'd'):
                subNetClass = subDenseNet
            elif(netType == 'Unet' or netType == 'u'):
                subNetClass = subUnet
            else:
                assert False, "no such subnetType:" + subnetType
            subNetClassList.append(subNetClass)
        # if(subnetType == 'Dense'):
        #     subNetClass = subDenseNet
        # elif(subnetType == 'Unet'):
        #     subNetClass = subUnet
        # else:
        #     assert False, "no such subnetType:"+subnetType
        templayerList = []
        for i in range(c):
            if(self.globalSkip):
                tmpSubNet = subNetClassList[i]((inChannel*(i+1),inChannel), fNum, growthRate, d, dilate, activation, useOri, transition, useSE)
            else:
                if(isinstance(inChannel,tuple)):
                    if(i==0):
                        tmpSubNet = subNetClassList[i](inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, useSE)
                    else:
                        tmpSubNet = subNetClassList[i](inChannel[1], fNum, growthRate, d, dilate, activation, useOri, transition, useSE)
                else:
                    tmpSubNet = subNetClassList[i](inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, useSE)
            tmpDF = dataConsistencyLayer_static(trick = trickList[i])
            templayerList.append(tmpSubNet)
            templayerList.append(tmpDF)
        if(globalResSkip):
            templayerList[-2].residual = False
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
        if(self.globalResSkip):
            xt = xt + x1
        
        return xt

class CN_Conv(nn.Module):
    def __init__(self, inChannel = 2, d = 5, c = 5, fNum = 32, dilate = False, trick = 0):
        '''
        trick 1 : abs->DC
        trick 2 : DC->abs->DC
        '''
        self.trick = trick
        super(CN_Conv, self).__init__()
        templayerList = []
        for i in range(c):
            tmpSubNet = convBlock(2, d, fNum, dilate)
            tmpDF = dataConsistencyLayer_static()
            templayerList.append(tmpSubNet)
            templayerList.append(tmpDF)
        self.layerList = nn.ModuleList(templayerList)

        if(self.trick in [3,4]):
            tmpConvList = []
            for i in range(1):
                tempConv = nn.Conv2d(4,2,1,padding=0)
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

class CN_SkipConv(nn.Module):
    def __init__(self, inChannel = 2, d = 5, c = 5, fNum = 32, dilate = False, trick = 0, skipMode = 0):
        super(CN_SkipConv, self).__init__()
        assert c==5, "only c = 5 is acceptable in skip mode"
        self.trick = trick
        self.skipMode = skipMode
        
        templayerList = []
        for i in range(c):
            if(i<2 or (i<3 and self.skipMode==0)):
                tmpSubNet = convBlock(2, d, fNum, dilate)
            else:
                tmpSubNet = convBlock(4, d, fNum, dilate)
            tmpDF = dataConsistencyLayer_static()
            templayerList.append(tmpSubNet)
            templayerList.append(tmpDF)
        self.layerList = nn.ModuleList(templayerList)

        if(self.trick in [3,4]):
            tmpConvList = []
            for i in range(c):
                tempConv = nn.Conv2d(4,2,1,padding=0)
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

class DualNetwork_HnL(nn.Module):
    def __init__(self, inChannel = 2, d = 5, c = 5, fNum = 16, growthRate = 16, dilate = False, activation = 'ReLU',  \
        useOri = False, transition = 0.5, trick = 0, globalDense = False, useSE = False, shareHL = False, fuseMode = 'Kspace'):
        super(DualNetwork_HnL, self).__init__()
        self.shareHL = shareHL
        self.fuseMode = fuseMode
        if(isinstance(c,int)):
            ch = c
            cl = c
            cf = c
        else:
            if(len(c)==2):
                ch = c[0]
                cl = c[0]
                cf = c[1]
            else:
                ch,cl,cf = c
        self.highPassNetwork = CN_Dense(inChannel,d,ch,fNum,growthRate,dilate,activation,useOri,transition,trick,globalDense,useSE)
        if(not self.shareHL):
            self.lowPassNetwork = CN_Dense(inChannel,d,cl,fNum,growthRate,dilate,activation,useOri,transition,trick,globalDense,useSE)
        if(self.fuseMode == 'Kspace'):
            self.fuseNetwork = CN_Dense(inChannel,d,cf,fNum,growthRate,dilate,activation,useOri,transition,trick,globalDense,useSE)
        else:
            self.infuseNetwork = CN_Dense((2*inChannel,inChannel),d,1,fNum,growthRate,dilate,activation,useOri,transition,trick,globalDense,useSE)
            self.fuseNetwork = CN_Dense(inChannel,d,cf-1,fNum,growthRate,dilate,activation,useOri,transition,trick,globalDense,useSE)

        highPassMask_np = gaussianFilter(256,256,64,True)
        lowPassMask_np = gaussianFilter(256,256,64,False)
        self.highPassMask = torch.from_numpy(highPassMask_np).type(torch.cuda.FloatTensor).reshape(1,256,256)
        self.lowPassMask = torch.from_numpy(lowPassMask_np).type(torch.cuda.FloatTensor).reshape(1,256,256)

    def getSubNetwork(self, index):
        if(index == 1):
            return self.highPassNetwork
        elif(index == 2):
            if(self.shareHL):
                return self.highPassNetwork
            else:
                return self.lowPassNetwork
        elif(index == 3):
            return self.fuseNetwork
        else:
            return self

    def kspaceFilter(self, x, highpass = True):
        if(highpass):
            return kspaceFilter(x, self.highPassMask)
        else:
            return kspaceFilter(x, self.lowPassMask)

    def forward(self, x1, y, mask, stage = 0):
        self.highPassMask = self.highPassMask.to(x1.device)
        self.lowPassMask = self.lowPassMask.to(x1.device)
        x_hp = kspaceFilter(x1,self.highPassMask)
        x_lp = kspaceFilter(x1,self.lowPassMask)
        y_hp = y * self.highPassMask.reshape(1,256,256,1)
        y_lp = y * self.lowPassMask.reshape(1,256,256,1)

        xg_hp = self.highPassNetwork(x_hp,y_hp,mask)
        if(stage == 1):
            return xg_hp
        if(self.shareHL):
            xg_lp = self.highPassNetwork(x_lp,y_lp,mask)
        else:
            xg_lp = self.lowPassNetwork(x_lp,y_lp,mask)
        if(stage == 2):
            return xg_lp
        #x_fuse = torch.cat([xg_hp[:,0:1],xg_lp[:,0:1]],1)
        if(self.fuseMode == 'Kspace'):
            x_fuse = kspaceFuse(xg_hp,xg_lp)
        else:
            x_fuse = torch.cat([xg_hp,xg_lp],1)
            x_fuse = self.infuseNetwork(x_fuse,y,mask)
        xg = self.fuseNetwork(x_fuse,y,mask)

        return xg

class DualNetwork_debug(nn.Module):
    def __init__(self, inChannel = 2, d = 5, c = 5, fNum = 16, growthRate = 16, dilate = False, activation = 'ReLU',  \
        useOri = False, transition = 0.5, trick = 0, globalDense = False, useSE = False):
        super(DualNetwork_debug, self).__init__()
        self.highPassNetwork = CN_Dense(inChannel,d,c,fNum,growthRate,dilate,activation,useOri,transition,trick,globalDense,useSE)
        self.lowPassNetwork = CN_Dense(inChannel,d,c,fNum,growthRate,dilate,activation,useOri,transition,trick,globalDense,useSE)
        self.fuseNetwork = CN_Dense(inChannel,d,c,fNum,growthRate,dilate,activation,useOri,transition,trick,globalDense,useSE)

        highPassMask_np = gaussianFilter(256,256,64,True)
        lowPassMask_np = gaussianFilter(256,256,64,False)
        self.highPassMask = torch.from_numpy(highPassMask_np).type(torch.cuda.FloatTensor).reshape(1,256,256)
        self.lowPassMask = torch.from_numpy(lowPassMask_np).type(torch.cuda.FloatTensor).reshape(1,256,256)

        self.dc = dataConsistencyLayer_static()


    def forward(self, x1, y, mask):
        self.highPassMask = self.highPassMask.to(x1.device)
        self.lowPassMask = self.lowPassMask.to(x1.device)
        x_hp = kspaceFilter(x1,self.highPassMask)
        x_lp = kspaceFilter(x1,self.lowPassMask)
        y_hp = y * self.highPassMask.reshape(1,256,256,1)
        y_lp = y * self.lowPassMask.reshape(1,256,256,1)

        xg_hp = self.dc(x_hp,y_hp,mask)
        xg_lp = self.dc(x_lp,y_hp,mask)
        # xg_hp = self.highPassNetwork(x_hp,y_hp,mask)
        # xg_lp = self.lowPassNetwork(x_lp,y_hp,mask)
        # x_fuse = torch.cat([xg_hp[:,0:1],xg_lp[:,0:1]],1)

        xg = kspaceFuse(xg_hp,xg_lp)
        # xg = self.fuseNetwork(x_fuse,y,mask)

        return xg