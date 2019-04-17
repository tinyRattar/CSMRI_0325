import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from collections import OrderedDict

from .networkUtil import *
from .CNN import vanillaCNN
#from .DenseUnet_dilate import DenseUnet_dilate
from .DenseUnet_origin import DUori_cn
from .RDN import RDN
from .RDN_complex import RDN_complex
from .DC_CNN import DC_CNN,DC_CNN_DDU
from .DC_CNN_dynamic import DC_CNN_dynamic_DS,DC_CNN_dynamic
from .DDU import DDU,DDU_c2,DDU_debug,DDU_cn
from .DDU_dynamic import DDU3d
#from .DDU_origin import DDU_ori
from .cascadeNetwork import CN_Dense,CN_Conv,CN_SkipConv
from .cascadeNetwork_dynamic import CN_Dense_3d,CN_Conv_3d

def getOptimizer(param, optimizerType, LR, weightDecay = 0):
    if(optimizerType == 'RMSprop'):
        optimizer = torch.optim.RMSprop(param, lr=LR)
    elif(optimizerType == 'Adam'):
        optimizer = torch.optim.Adam(param, lr=LR)
    elif(optimizerType == 'Adam_wd'):
        optimizer = torch.optim.Adam(param, lr=LR, weight_decay = weightDecay)
    elif(optimizerType == 'Adam_RDN'):
        optimizer = torch.optim.Adam(param, lr=LR, weight_decay = 0.0001) #weight decay for RDN
    elif(optimizerType == 'Adam_DC_CNN'):
        optimizer = torch.optim.Adam(param, lr=LR, weight_decay = 1e-7) #weight decay for DC_CNN
    else:
        assert False,"Wrong optimizer type"
        
    return optimizer

def getNet(netType):
    if('3d' in netType):
        return getNet3d(netType[3:])
    if(netType == 'DDU'):
        return DDU(1)
    elif(netType == 'DDU_leakyReLU'):
        return DDU(1,'LeakyReLU')
    elif(netType == 'DDU_c2'):
        return DDU_c2(1)
    elif(netType == 'DDU_c2_complex'):
        return DDU_c2(1, isComplex = True)
    elif(netType == 'DDU_debug'):
        return DDU_debug(1)
    elif(netType == 'DDU_c5'):
        return DDU_cn(1,c=5)
    elif(netType == 'DDU_c2_addon'):
        return DDU_cn(1, c=2,addon=True)
    #===========DenseUnet-origin==
    elif(netType == 'DDUori_c2'):
        return DUori_cn(1, dilate=True, c=2)
    #===========DC_CNN============
    elif(netType == 'DCCNN'):#41.02
        return DC_CNN()
    elif(netType == 'DCCNN_f16'):#40.81
        return DC_CNN(fNum=16)
    elif(netType == 'DCCNN_f64'):#41.19
        return DC_CNN(fNum=64)
    elif(netType == 'DCCNN-2d2c'):
        return DC_CNN(2,2)
    elif(netType == 'DCCNN_DDU-2c'):
        return DC_CNN_DDU(2)
    #===========RDN===============
    elif(netType == 'RDN'):
        return RDN()
    elif(netType == 'RDN_DC'):
        return RDN(dcLayer = 'DC')
    elif(netType == 'RDN_complex'):
        return RDN_complex()
    elif(netType == 'RDN_complex_DC'):
        return RDN_complex(dcLayer = 'DC')
    #===========cascade===========
    #-----------DD-------------
    elif(netType == 'CN_DD_c5_complex'):
        return CN_Dense(2,c=5,dilate=True)
    elif(netType == 'CN_DD_c5_complex_ori'):
        return CN_Dense(2,c=5,dilate=True, useOri = True)
    elif(netType == 'CN_DD_c5_complex_f32'):
        return CN_Dense(2,c=5,dilate=True,fNum = 32)
    elif(netType == 'CN_DD_c5_complex_fg32'):
        return CN_Dense(2,c=5,dilate=True,fNum = 32,growthRate = 32)
    elif(netType == 'CN_DD_c2_complex'):
        return CN_Dense(2,c=2,dilate=True)
    #-----------Dense----------
    elif(netType == 'CN_Dense_c5_complex'):
        return CN_Dense(2,c=5,dilate=False)
    elif(netType == 'CN_Dense_c5_complex_tr'):
        return CN_Dense(2, c=5, dilate=False, transition=0.5)
    #-----------DenseOri-------
    elif(netType == 'CN_Ori_c5_complex'):
        return CN_Dense(2,c=5,dilate=False, useOri = True)
    elif(netType == 'CN_Ori_c5_complex_tr'):
        return CN_Dense(2,c=5,dilate=False, useOri = True, transition=0.5)
    elif(netType == 'CN_Ori_c5_complex_tr_se'):
        return CN_Dense(2,c=5,dilate=False, useOri = True, transition=0.5, useSE = True)
    elif(netType == 'CN_Ori_c5_complex_tr_trick2'):
        return CN_Dense(2,c=5,dilate=False, useOri = True, transition=0.5, trick = 2)
    elif(netType == 'CN_Ori_c5_complex_tr_trick4'):
        return CN_Dense(2,c=5,dilate=False, useOri = True, transition=0.5, trick = 4)
    elif(netType == 'CN_Ori_c5_complex_f32'):
        return CN_Dense(2,c=5,dilate=False, fNum = 32, useOri = True)
    elif(netType == 'CN_Ori_c5_complex_f32_tr'):
        return CN_Dense(2,c=5,dilate=False, fNum = 32, useOri = True, transition=0.5)
    elif(netType == 'CN_Ori_c5_complex_fg32'):
        return CN_Dense(2,c=5,dilate=False, fNum = 32, growthRate = 32, useOri = True)
    elif(netType == 'CN_Ori_c5_complex_fg32_tr'):
        return CN_Dense(2,c=5,dilate=False, fNum = 32, growthRate = 32, useOri = True, transition=0.5)
    #-----------dOri-----------
    elif(netType == 'CN_dOri_c5_complex'): # same with 'CN_DD_c5_complex_ori'
        return CN_Dense(2,c=5,dilate=True, useOri = True)
    elif(netType == 'CN_dOri_c5_complex_tr'): 
        return CN_Dense(2,c=5,dilate=True, useOri = True, transition=0.5)
    elif(netType == 'CN_dOri_c5_complex_f32_tr'): 
        return CN_Dense(2,c=5,fNum = 32, dilate=True, useOri = True, transition=0.5)
    elif(netType == 'CN_dOri_c5_complex_fg32_tr'): 
        return CN_Dense(2,c=5,fNum = 32, growthRate = 32, dilate=True, useOri = True, transition=0.5)
    elif(netType == 'CN_dOri_c5_complex_tr_trick4'): 
        return CN_Dense(2,c=5,dilate=True, useOri = True, transition=0.5, trick = 4)
    #-----------globalDense----
    elif(netType == 'CN_Ori_c5_complex_tr_gd'):
        return CN_Dense(2,c=5,dilate=False, useOri = True, transition=0.5, globalDense = True)
    #-----------Conv-----------
    elif(netType == 'CN_Conv_c5_complex'):
        return CN_Conv(2,c=5,dilate=False)
    elif(netType == 'CN_Conv_c5_complex_trick1'):
        return CN_Conv(2,c=5,trick = 1)
    elif(netType == 'CN_Conv_c5_complex_trick2'):
        return CN_Conv(2,c=5,trick = 2)
    elif(netType == 'CN_Conv_c5_complex_trick3'):
        return CN_Conv(2,c=5,trick = 3)
    elif(netType == 'CN_Conv_c5_complex_trick4'):
        return CN_Conv(2,c=5,trick = 4)
    elif(netType == 'CN_Conv_c5_complex_dilate'):
        return CN_Conv(2,c=5,dilate=True)
    #-----------SkipConv-------
    elif(netType == 'CN_SkipConv_c5_complex'):
        return CN_SkipConv(2,c=5,skipMode = 0)
    elif(netType == 'CN_SkipConv_c5_complex_sm1'):
        return CN_SkipConv(2,c=5,skipMode = 1)
    
    #===========Others============
    elif(netType == 'vanillaCNN'):
        return vanillaCNN()
    else:
        assert False,"Wrong net type"

def getNet3d(netType):
    if(netType == 'DCCNN'):
        return DC_CNN_dynamic()
    elif(netType == 'DCCNN_DS'):
        return DC_CNN_dynamic_DS()
    #-----------Conv-----------
    elif(netType == 'CN_Conv_c5_complex'):
        return CN_Conv_3d(2,c=5,dilate=False)
    elif(netType == 'CN_Conv_c5_complex_trick2'):
        return CN_Conv_3d(2,c=5,trick = 2)
    elif(netType == 'CN_Conv_c5_complex_trick4'):
        return CN_Conv_3d(2,c=5,trick = 4)
    #-----------DenseOri-------
    elif(netType == 'CN_Ori_c5_complex_tr'):
        return CN_Dense_3d(2,c=5,dilate=False, useOri = True, transition=0.5)
    elif(netType == 'CN_Ori_c5_complex_tr_trick2'):
        return CN_Dense_3d(2,c=5,dilate=False, useOri = True, transition=0.5, trick = 2)
    elif(netType == 'CN_Ori_c5_complex_tr_trick4'):
        return CN_Dense_3d(2,c=5,dilate=False, useOri = True, transition=0.5, trick = 4)
    #-----------dOri-----------
    elif(netType == 'CN_dOri_c5_complex_tr'): 
        return CN_Dense_3d(2,c=5,dilate=True, useOri = True, transition=0.5)
    elif(netType == 'CN_dOri_c5_complex_f32_tr'): 
        return CN_Dense_3d(2,c=5,fNum = 32, dilate=True, useOri = True, transition=0.5)
    elif(netType == 'CN_dOri_c5_complex_fg32_tr'): 
        return CN_Dense_3d(2,c=5,fNum = 32, growthRate = 32, dilate=True, useOri = True, transition=0.5)
    elif(netType == 'CN_dOri_c5_complex_tr_trick4'): 
        return CN_Dense_3d(2,c=5,dilate=True, useOri = True, transition=0.5, trick = 4)
    #===========Others============
    else:
        assert False,"Wrong net type"
        
def getNet_2net(netType,index):
    if(netType == 'DDU'):
        if(index==0):
            return DDU(1)
        else:
            return dataConsistencyLayer()
    elif(netType == 'DDU3d_leakyReLU'):
        if(index==0):
            return DilatedDenseUnet3d('LeakyReLU')
        else:
            return dataConsistencyLayer()
    else:
        assert False,"Wrong net type"

def getNet_list(netType):
    if(netType == 'DDU'):
        listNet = [DDU(1), dataConsistencyLayer()]
    else:
        assert False,"Wrong net type"

    return nn.ModuleList(listNet)

def getLoss(lossType):
    if(lossType == 'mse'):
        return torch.nn.MSELoss()
    elif(lossType == 'mae'):
        return torch.nn.L1Loss()
    else:
        assert False,"Wrong loss type"