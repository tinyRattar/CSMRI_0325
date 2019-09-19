import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from collections import OrderedDict

from .networkUtil import *
from .CNN import vanillaCNN,Unet_dc
from .RDN import RDN
from .RDN_complex import RDN_complex
from .DC_CNN import DC_CNN
from .cascadeNetwork import CN_Dense,CN_Conv

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
    #===========DC_CNN============
    if(netType == 'DCCNN'):
        return DC_CNN()
    elif(netType == 'DCCNN_f16'):
        return DC_CNN(fNum=16)
    elif(netType == 'DCCNN_f64'):
        return DC_CNN(fNum=64)
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
    elif(netType == 'CN_Ori_d7c5_complex_tr'):
        return CN_Dense(2,d=7,c=5,dilate=False, useOri = True, transition=0.5)
    elif(netType == 'CN_Ori_c10_complex_tr'):
        return CN_Dense(2,c=10,dilate=False, useOri = True, transition=0.5)
    #-----------dOri-----------
    elif(netType == 'CN_dOri_c5_complex'):
        return CN_Dense(2,c=5,dilate=True, useOri = True)
    elif(netType == 'CN_dOri_c5_complex_tr'): 
        return CN_Dense(2,c=5,dilate=True, useOri = True, transition=0.5)
    elif(netType == 'CN_dOri_c5_complex_f32_tr'): 
        return CN_Dense(2,c=5,fNum = 32, dilate=True, useOri = True, transition=0.5)
    elif(netType == 'CN_dOri_c5_complex_fg32_tr'): 
        return CN_Dense(2,c=5,fNum = 32, growthRate = 32, dilate=True, useOri = True, transition=0.5)
    elif(netType == 'CN_dOri_c5_complex_tr_trick2'):
        return CN_Dense(2,c=5,dilate=True, useOri = True, transition=0.5, trick = 2)
    elif(netType == 'CN_dOri_c5_complex_tr_trick4'):
        return CN_Dense(2,c=5,dilate=True, useOri = True, transition=0.5, trick = 4)
    elif(netType == 'CN_dOri_c5_complex_tr_trick2n4'):
        return CN_Dense(2,c=5,dilate=True, useOri = True, transition=0.5, trick = [2, 2, 2, 2, 4])
    elif(netType == 'CN_dOri_c5_complex_tr_trick4_gr'):
        return CN_Dense(2,c=5,dilate=True, useOri = True, transition=0.5, trick = 4, globalResSkip = True)
    elif(netType == 'CN_dOri_c10_complex_tr_trick2'):
        return CN_Dense(2,c=10,dilate=True, useOri = True, transition=0.5, trick = 2)
    elif(netType == 'CN_dOri_c10_complex_tr_trick4'):
        return CN_Dense(2,c=10,dilate=True, useOri = True, transition=0.5, trick = 4)
    elif(netType == 'CN_dOri_c10_complex_tr_trick4_se'):
        return CN_Dense(2,c=10,dilate=True, useOri = True, transition=0.5, trick = 4, useSE = True)
    #-----------cascadeNumber----
    elif(netType == 'CN_dOri_c1_complex_tr_trick2'):
        return CN_Dense(2,c=1,dilate=True, useOri = True, transition=0.5, trick = 2)
    elif(netType == 'CN_dOri_c2_complex_tr_trick2'):
        return CN_Dense(2,c=2,dilate=True, useOri = True, transition=0.5, trick = 2)
    elif(netType == 'CN_dOri_c3_complex_tr_trick2'):
        return CN_Dense(2,c=3,dilate=True, useOri = True, transition=0.5, trick = 2)
    elif(netType == 'CN_dOri_c4_complex_tr_trick2'):
        return CN_Dense(2,c=4,dilate=True, useOri = True, transition=0.5, trick = 2)
    elif(netType == 'CN_dOri_c6_complex_tr_trick2'):
        return CN_Dense(2,c=6,dilate=True, useOri = True, transition=0.5, trick = 2)
    elif(netType == 'CN_dOri_c7_complex_tr_trick2'):
        return CN_Dense(2,c=7,dilate=True, useOri = True, transition=0.5, trick = 2)
    elif(netType == 'CN_dOri_c8_complex_tr_trick2'):
        return CN_Dense(2,c=8,dilate=True, useOri = True, transition=0.5, trick = 2)
    elif(netType == 'CN_dOri_c9_complex_tr_trick2'):
        return CN_Dense(2,c=9,dilate=True, useOri = True, transition=0.5, trick = 2)
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
    elif(netType == 'Unet_dc'):
        return Unet_dc()
    else:
        assert False,"Wrong net type"

def getLoss(lossType):
    if(lossType == 'mse'):
        return torch.nn.MSELoss()
    elif(lossType == 'mae'):
        return torch.nn.L1Loss()
    else:
        assert False,"Wrong loss type"