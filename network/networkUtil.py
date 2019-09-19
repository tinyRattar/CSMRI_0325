import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class denseBlockLayer(nn.Module):
    def __init__(self,inChannel=64, outChannel=64, kernelSize=3, inception = False, dilateScale = 1, activ = 'ReLU'):
        super(denseBlockLayer, self).__init__()
        self.useInception = inception

        if(self.useInception):
            self.conv1 = nn.Conv2d(inChannel,outChannel,3,padding = 1)
            self.conv2 = nn.Conv2d(inChannel,outChannel,5,padding = 2)
            self.conv3 = nn.Conv2d(inChannel,outChannel,7,padding = 3)
            if(activ == 'LeakyReLU'):
                self.relu = nn.LeakyReLU()
            else:
                self.relu = nn.ReLU()
            self.conv4 = nn.Conv2d(outChannel*3,outChannel,1,padding = 0)
            #self.relu2 = nn.ReLU()
        else:
            pad = int(dilateScale * (kernelSize - 1) / 2)
            
            self.conv = nn.Conv2d(inChannel,outChannel,kernelSize,padding = pad, dilation = dilateScale)
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
    
class denseBlock(nn.Module):
    def __init__(self, inChannel=64, outChannel=64, kernelSize=3, growthRate=16, layer=4, inceptionLayer = False, dilationLayer = False, activ = 'ReLU'):
        super(denseBlock, self).__init__()
        dilate = 1
        if(dilationLayer):
            dilateMulti = 2
        else:
            dilateMulti = 1
        pad = int((kernelSize-1)/2)
        self.layer = layer
        templayerList = []
        for i in range(0, layer):
            tempLayer = denseBlockLayer(inChannel+growthRate*i,growthRate,kernelSize,inceptionLayer,dilate,activ)
            dilate = dilate * dilateMulti
            templayerList.append(tempLayer)
        self.layerList = nn.ModuleList(templayerList)
        self.outputLayer = denseBlockLayer(inChannel+growthRate*layer,outChannel,kernelSize,inceptionLayer,1,activ)
        self.bn = nn.BatchNorm2d(outChannel)
            
    def forward(self,x):
        for i in range(0, self.layer):
            tempY = self.layerList[i](x)
            x = torch.cat((x, tempY), 1)
        y = self.outputLayer(x)
        y = self.bn(y)
            
        return y
    
class SELayer(nn.Module):
    def __init__(self,channel,height,width):
        super(SELayer, self).__init__()
        self.inShape = [channel,height,width]
        self.globalPooling = nn.AvgPool2d((self.inShape[1],self.inShape[2]))
        self.fc1 = nn.Conv2d(self.inShape[0],int(self.inShape[0]/2),1)
        self.fc2 = nn.Conv2d(int(self.inShape[0]/2),self.inShape[0],1)
    
    def forward(self,x):
        v1 = self.globalPooling(x)
        v2 = self.fc1(v1)
        v3 = self.fc2(v2)
        
        f = x*v3
        
        return f
    
class denseBlockLayer_origin(nn.Module):
    def __init__(self,inChannel=64, outChannel=64, kernelSize = 3, bottleneckChannel = 64, dilateScale = 1, activ = 'ReLU'):
        super(denseBlockLayer_origin, self).__init__()
        pad = int((kernelSize-1)/2)

        self.bn = nn.BatchNorm2d(inChannel)
        if(activ == 'LeakyReLU'):
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inChannel,bottleneckChannel,1)
        
        self.bn2 = nn.BatchNorm2d(bottleneckChannel)
        if(activ == 'LeakyReLU'):
            self.relu2 = nn.LeakyReLU()
        else:
            self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(bottleneckChannel,outChannel,kernelSize,padding = dilateScale,dilation=dilateScale)
        
            
    def forward(self,x):
        x1 = self.bn(x)
        x1 = self.relu(x1)
        x1 = self.conv(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        y = self.conv2(x1)
            
        return y
    
class denseBlock_origin(nn.Module):
    def __init__(self, inChannel=64, kernelSize=3, growthRate=16, layer=4, bottleneckMulti = 4, dilationLayer = False, activ = 'ReLU'):
        super(denseBlock_origin, self).__init__()
        dilate = 1
        if(dilationLayer):
            dilateMulti = 2
        else:
            dilateMulti = 1
        pad = int((kernelSize-1)/2)
        self.layer = layer
        templayerList = []
        for i in range(0, layer):
            tempLayer = denseBlockLayer_origin(inChannel+growthRate*i,growthRate,kernelSize,bottleneckMulti*growthRate, dilate, activ)
            dilate = dilate * dilateMulti
            templayerList.append(tempLayer)
        self.layerList = nn.ModuleList(templayerList)
            
    def forward(self,x):
        for i in range(0, self.layer):
            tempY = self.layerList[i](x)
            x = torch.cat((x, tempY), 1)
            
        return x
    
class transitionLayer(nn.Module):
    def __init__(self, inChannel = 64, compressionRate = 0.5, activ = 'ReLU'):
        super(transitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(inChannel)
        if(activ == 'LeakyReLU'):
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inChannel,int(inChannel*compressionRate),1)
        
    def forward(self,x):
        x1 = self.bn(x)
        x2 = self.relu(x1)
        y = self.conv(x2)
        
        return y

class convLayer(nn.Module):
    def __init__(self, inChannel = 64, outChannel = 64, activ = 'ReLU', kernelSize = 3):
        super(convLayer, self).__init__()
        self.bn = nn.BatchNorm2d(inChannel)
        pad = int((kernelSize-1)/2)
        if(activ == 'LeakyReLU'):
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inChannel,outChannel,kernelSize,padding = pad)
        
    def forward(self,x):
        x1 = self.bn(x)
        x2 = self.relu(x1)
        y = self.conv(x2)
        
        return y
    
class dataConsistencyLayer(nn.Module):
    def __init__(self, initLamda = 1, isStatic = False):
        super(dataConsistencyLayer, self).__init__()
        self.normalized = True #norm == 'ortho'
        self.lamda = Parameter(torch.Tensor(1))
        self.lamda.data.uniform_(0, 1)
        self.isStatic = isStatic
    
    def forward(self, xin, y, mask):
        if(self.isStatic):
            iScale = 1
        else:
            iScale = self.lamda/(1+self.lamda)
        if(len(xin.shape)==4):
            if(xin.shape[1]==1):
                emptyImag = torch.zeros_like(xin)
                xin_c = torch.cat([xin,emptyImag],1).permute(0,2,3,1)
            else:
                xin_c = xin.permute(0,2,3,1)
            mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],1)
        elif(len(xin.shape)==5):
            if(xin.shape[1]==1):
                emptyImag = torch.zeros_like(xin)
                xin_c = torch.cat([xin,emptyImag],1).permute(0,2,3,4,1)
            else:
                xin_c = xin.permute(0,2,3,4,1)
            mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],mask.shape[3],1)
        else:
            assert False, "xin shape length has to be 4(2d) or 5(3d)"
        
        xin_f = torch.fft(xin_c,2, normalized=self.normalized)
        xGT_f = y
        
        xout_f = xin_f + (- xin_f + xGT_f) * iScale * mask

        xout = torch.ifft(xout_f,2, normalized=self.normalized)
        if(len(xin.shape)==4):
            xout = xout.permute(0,3,1,2)
        else:
            xout = xout.permute(0,4,1,2,3)
        if(xin.shape[1]==1):
            xout = torch.sqrt(xout[:,0:1]*xout[:,0:1]+xout[:,1:2]*xout[:,1:2])
        
        return xout

class dataConsistencyLayer_static(nn.Module):
    def __init__(self, initLamda = 1, trick = 0, dynamic = False, conv = None):
        super(dataConsistencyLayer_static, self).__init__()
        self.normalized = True 
        self.trick = trick
        tempConvList = []
        if(self.trick in [3,4]):
            if(conv is None):
                if(dynamic):
                    conv = nn.Conv3d(4,2,1,padding=0)
                else:
                    conv = nn.Conv2d(4,2,1,padding=0)
            tempConvList.append(conv)
        self.trickConvList = nn.ModuleList(tempConvList)

    def dc_operate(self, xin, y, mask):
        iScale = 1
        if(len(xin.shape)==4):
            if(xin.shape[1]==1):
                emptyImag = torch.zeros_like(xin)
                xin_c = torch.cat([xin,emptyImag],1).permute(0,2,3,1)
            else:
                xin_c = xin.permute(0,2,3,1)
            mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],1)
        elif(len(xin.shape)==5):
            if(xin.shape[1]==1):
                emptyImag = torch.zeros_like(xin)
                xin_c = torch.cat([xin,emptyImag],1).permute(0,2,3,4,1)
            else:
                xin_c = xin.permute(0,2,3,4,1)
            mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],mask.shape[3],1)
        else:
            assert False, "xin shape length has to be 4(2d) or 5(3d)"
        
        xin_f = torch.fft(xin_c,2, normalized=self.normalized)
        xGT_f = y
        
        xout_f = xin_f + (- xin_f + xGT_f) * iScale * mask

        xout = torch.ifft(xout_f,2, normalized=self.normalized)
        if(len(xin.shape)==4):
            xout = xout.permute(0,3,1,2)
        else:
            xout = xout.permute(0,4,1,2,3)
        if(xin.shape[1]==1):
            xout = torch.sqrt(xout[:,0:1]*xout[:,0:1]+xout[:,1:2]*xout[:,1:2])
        
        return xout
    
    def forward(self, xin, y, mask):
        xt = xin
        if(self.trick == 1):
            xt = abs4complex(xt)
            xt = self.dc_operate(xt, y, mask)
        elif(self.trick == 2):
            xt = self.dc_operate(xt, y, mask)
            xt = abs4complex(xt)
            xt = self.dc_operate(xt, y, mask)
        elif(self.trick == 3):
            xdc1 = self.dc_operate(xt, y, mask)
            xt = abs4complex(xt)
            xdc2 = self.dc_operate(xt, y, mask)
            xdc = torch.cat([xdc1,xdc2],1)
            xt = self.trickConvList[0](xdc)
        elif(self.trick == 4):
            xdc1 = self.dc_operate(xt, y, mask)
            xabs = abs4complex(xdc1)
            xdc2 = self.dc_operate(xabs, y, mask)
            xdc = torch.cat([xdc1,xdc2],1)
            xt = self.trickConvList[0](xdc)
        else:
            xt = self.dc_operate(xt, y, mask)

        return xt
        

def abs4complex(x):
    y = torch.zeros_like(x)
    y[:,0:1] = torch.sqrt(x[:,0:1]*x[:,0:1]+x[:,1:2]*x[:,1:2])
    y[:,1:2] = 0

    return y

def kspaceFilter(xin, mask):
    if(len(xin.shape)==4):
        if(xin.shape[1]==1):
            emptyImag = torch.zeros_like(xin)
            xin_c = torch.cat([xin,emptyImag],1).permute(0,2,3,1)
        else:
            xin_c = xin.permute(0,2,3,1)
        mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],1)
    elif(len(xin.shape)==5):
        if(xin.shape[1]==1):
            emptyImag = torch.zeros_like(xin)
            xin_c = torch.cat([xin,emptyImag],1).permute(0,2,3,4,1)
        else:
            xin_c = xin.permute(0,2,3,4,1)
        assert False,"3d not support"
    else:
        assert False, "xin shape length has to be 4(2d) or 5(3d)"
    
    xin_f = torch.fft(xin_c,2, normalized=True)
    
    xout_f = xin_f*mask

    xout = torch.ifft(xout_f,2, normalized=True)
    if(len(xin.shape)==4):
        xout = xout.permute(0,3,1,2)
    else:
        xout = xout.permute(0,4,1,2,3)
    if(xin.shape[1]==1):
        xout = torch.sqrt(xout[:,0:1]*xout[:,0:1]+xout[:,1:2]*xout[:,1:2])
    
    return xout

def kspaceFuse(x1,x2):
    lout = []
    for xin in [x1,x2]:
        if(len(xin.shape)==4):
            if(xin.shape[1]==1):
                emptyImag = torch.zeros_like(xin)
                xin_c = torch.cat([xin,emptyImag],1).permute(0,2,3,1)
            else:
                xin_c = xin.permute(0,2,3,1)
        elif(len(xin.shape)==5):
            if(xin.shape[1]==1):
                emptyImag = torch.zeros_like(xin)
                xin_c = torch.cat([xin,emptyImag],1).permute(0,2,3,4,1)
            else:
                xin_c = xin.permute(0,2,3,4,1)
        else:
            assert False, "xin shape length has to be 4(2d) or 5(3d)"
        lout.append(xin_c)
    x1c,x2c = lout
    
    x1f = torch.fft(x1c,2, normalized=True)
    x2f = torch.fft(x2c,2, normalized=True)

    xout_f = x1f+x2f

    xout = torch.ifft(xout_f,2, normalized=True)
    if(len(x1.shape)==4):
        xout = xout.permute(0,3,1,2)
    else:
        xout = xout.permute(0,4,1,2,3)
    if(xin.shape[1]==1):
        xout = torch.sqrt(xout[:,0:1]*xout[:,0:1]+xout[:,1:2]*xout[:,1:2])

    return xout