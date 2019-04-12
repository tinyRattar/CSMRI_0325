import numpy as np
from PIL import Image
import torch.utils.data as data
import scipy.io as sio
import random

from util.imageUtil import *

fakeRandomPath = 'mask/mask_r30k_29.mat'
fakeRandomPath_15 = 'mask/mask_r10k_15.mat'

def generateDatasetName(configs):
    datasetName = ""
    for index in range(len(configs)):
        if(configs[index]==""):
            continue
        datasetName += configs[index]
        datasetName += '_'

    return datasetName[:-1]

def getDataloader(dataType = '1in1',mode = 'train', batchSize = 1):
    pathDirData = 'data/cardiac_ktz/'
    if(mode == 'train'):
        shuffleFlag = True
        a = 1
        b = 31
    else:
        shuffleFlag = False
        a = 31
        b = 34
    if('reduce' in dataType):
        reduceMode = "reduce"
    else:
        reduceMode = ""
    #=============================
    if('1in1old' in dataType):
        mainType = '1in1old'
    elif('1in1' in dataType):
        mainType = '1in1'
    elif('3in1' in dataType):
        mainType = '3in1'
    elif('3d' in dataType):
        mainType = '3d'
    else:
        assert False,"DataType ERROR: No MainType Include"
    #-----------------------
    if('complex' in dataType):
        dataMode = 'complex'
    elif('abs' in dataType):
        dataMode = 'abs'
    else:
        dataMode = 'abs'
    #-----------------------
    if(('fakeRandom' in dataType) or ('random' in dataType)):
        samplingMode = 'fakeRandom'
    elif(('r15' in dataType) or ('rand15' in dataType)):
        samplingMode = 'rand15'
    elif('nolattice' in dataType):
        samplingMode = 'nolattice'
    elif('static' in dataType):
        samplingMode = 'static'
    else:
        samplingMode = 'lattice'
        #assert False,"DataType ERROR: No samplingMode Include"
    #-----------------------
    if('npatch8' in dataType):
        npatch = 8
        strNpatch = "npatch8"
    else:
        npatch = 1
        strNpatch = ""
    datasetName = generateDatasetName([mainType,dataMode,strNpatch,samplingMode,mode,reduceMode])
    print('#Generating dataset:'+datasetName)
    #=============================
    if(mainType == '1in1'):
        #dataset = dataset_xin1(a, b, dataMode, samplingMode, 1, reduceMode)
        dataset = dataset_1in1_noImg(a, b, dataMode, samplingMode, reduceMode)
    elif(mainType == '1in1old'):
        dataset = dataset_xin1(a, b, dataMode, samplingMode, 1, reduceMode)
    elif(mainType == '3in1'):
        dataset = dataset_xin1(a, b, dataMode, samplingMode, 3, reduceMode)
    elif(mainType == '3d'):
        dataset = dataset_3d_noImg(a, b, dataMode, samplingMode, npatch)
    else:
        assert False,"wrong dataset type"
        
    #flag = (mode == 'train')
    data_loader = data.DataLoader(dataset, batch_size=batchSize, shuffle=shuffleFlag)
    datasize = len(dataset)
    return data_loader,datasize

class dataset_3d(data.Dataset):
    def __init__(self, iStart = 1,iEnd = 31, mode = 'abs', samplingMode = 'default', npatch = 1):
        if(samplingMode == 'fakeRandom'):
            mDic = sio.loadmat(fakeRandomPath)
            miList = mDic['RAll']
        elif(samplingMode == 'rand15'):
            mDic = sio.loadmat(fakeRandomPath_15)
            miList = mDic['RAll']
        self.xList = []
        self.yList = []
        self.mList = []
        #self.mode = mode
        offset=0
        for i in range(iStart,iEnd):
            for z in range(0,5): #0,4
                for p in range(npatch):
                    x = np.zeros((1,20,256,int(256/npatch)))
                    y = np.zeros((1,20,256,int(256/npatch)))
                    if(mode == 'complex'):
                        x = np.zeros((2,20,256,int(256/npatch)))
                        y = np.zeros((2,20,256,int(256/npatch)))
                    m = np.zeros((20,256,int(256/npatch)))
                    for t in range(1,21): #1,20
                        filename = "data/cardiac_ktz/mr_heart_p%02dt%02dz%d.png" % (i,t,z)
                        im = Image.open(filename)
                        im_np = np.array(im).astype(np.float32)/255.
                        
                        if(samplingMode == 'fakeRandom' or ('rand' in samplingMode)):
                            randI = random.randrange(miList.shape[1])
                            mi = miList[:,randI]
                            subF,mask = kspace_subsampling(im_np, 0, 'fakeRandom', mi)
                        else:
                            subF,mask = kspace_subsampling(im_np, offset)
                        subImg = f2img(subF)[:,p*int(256/npatch):(p+1)*int(256/npatch)]
                        mask = np.fft.ifftshift(mask)

                        if(mode == 'abs'):
                            x[0,t-1] = np.abs(subImg)
                        elif(mode == 'complex'):
                            x[0,t-1] = np.real(subImg)
                            x[1,t-1] = np.imag(subImg)
                        else:
                            assert False,"real mode is abandoned"
                            x[0,t-1] = np.real(subImg)
                        y[0,t-1] = im_np[:,p*int(256/npatch):(p+1)*int(256/npatch)]
                        m[t-1] = mask[:,p*int(256/npatch):(p+1)*int(256/npatch)]
                        
                        if(samplingMode == 'nolattice'):
                            pass
                        else:
                            offset = (offset+1)%4
                    self.xList.append(x)
                    self.yList.append(y)
                    self.mList.append(m)

    def __getitem__(self, index):
        img = self.xList[index]
        label = self.yList[index]
        mask = self.mList[index]
        
        return img, label, mask

    def __len__(self):
        return len(self.xList)
    
class dataset_xin1(data.Dataset):
    def __init__(self, iStart = 1,iEnd = 31, mode = 'abs', samplingMode = 'default', xin1 = 1, reduceMode = ""):
        assert xin1 == 1, "xin1 not support now"
        self.xin1 = xin1
        if(samplingMode == 'fakeRandom'):
            mDic = sio.loadmat(fakeRandomPath)
            miList = mDic['RAll']
        if(mode == 'complex'):
            self.xList = []
            self.yList = []
        else:
            self.xList = []
            self.yList = []
        self.mList = []
        offset = 0
        index = 0
        if(reduceMode == "reduce"):
            tList = [1,5,15,20]
        else:
            tList = range(1,21)
        if('nolattice_offset' in samplingMode):
            offset = int(samplingMode[-1])
        for i in range(iStart,iEnd): 
            for z in range(0,5): #0,4
                for t in tList: #1,20
                    #index = (i-iStart)*100+z*20+t-1
                    filename = "data/cardiac_ktz/mr_heart_p%02dt%02dz%d.png" % (i,t,z)
                    im = Image.open(filename)
                    im_np = np.array(im).astype(np.float32)/255.
                    
                    if(samplingMode == 'fakeRandom'):
                        randI = random.randrange(miList.shape[1])
                        mi = miList[:,randI]
                        subF,mask = kspace_subsampling(im_np, 0, 'fakeRandom', mi)
                    else:
                        subF,mask = kspace_subsampling(im_np, offset)
                    subImg = f2img(subF)
                    m = np.fft.ifftshift(mask)
                    
                    if(mode == 'abs'):
                        x = np.zeros((1,256,256))
                        y = np.zeros((1,256,256))
                        x[0] = np.abs(subImg)
                        #m = mask
                    elif(mode == 'complex'):
                        x = np.zeros((2,256,256))
                        y = np.zeros((2,256,256))
                        #m = np.zeros((1,256,256))
                        x[0] = np.real(subImg)
                        x[1] = np.imag(subImg)
                        #m = mask
                    else:
                        assert False,"real mode is abandoned"
                        x[0] = np.real(subImg)
                    #m = mask
                    y[0] = im_np
                    
                    self.xList.append(x)
                    self.yList.append(y)
                    self.mList.append(m)
                    
                    if(samplingMode == 'nolattice'):
                        pass
                    else:
                        offset = (offset+1)%4

                    index += 1

    def __getitem__(self, index):
        i = index
        img = self.xList[i]
        label = self.yList[i]
        mask = self.mList[i]
        
        return img, label, mask

    def __len__(self):
        return len(self.xList)

class dataset_3d_noImg(data.Dataset):
    def __init__(self, iStart = 1,iEnd = 31, mode = 'abs', samplingMode = 'default', npatch = 1):
        if(samplingMode == 'fakeRandom'):
            mDic = sio.loadmat(fakeRandomPath)
            miList = mDic['RAll']
        elif(samplingMode == 'rand15'):
            mDic = sio.loadmat(fakeRandomPath_15)
            miList = mDic['RAll']
        self.yList = []
        self.mList = []
        assert mode in ['complex','abs'], "real mode is abandoned"
        self.mode = mode
        offset=0
        for i in range(iStart,iEnd):
            for z in range(0,5): #0,4
                for p in range(npatch):
                    #x = np.zeros((1,20,256,int(256/npatch)))
                    y = np.zeros((1,20,256,int(256/npatch)))
                    if(mode == 'complex'):
                        #x = np.zeros((2,20,256,int(256/npatch)))
                        y = np.zeros((2,20,256,int(256/npatch)))
                    m = np.zeros((20,256,int(256/npatch)))
                    for t in range(1,21): #1,20
                        filename = "data/cardiac_ktz/mr_heart_p%02dt%02dz%d.png" % (i,t,z)
                        im = Image.open(filename)
                        im_np = np.array(im).astype(np.float32)/255.
                        
                        if(samplingMode == 'fakeRandom' or ('rand' in samplingMode)):
                            randI = random.randrange(miList.shape[1])
                            mi = miList[:,randI]
                            mask = subsampling_mask(im_np, 0, 'fakeRandom', mi)
                        else:
                            mask = subsampling_mask(im_np, offset)
                        mask = np.fft.ifftshift(mask)

                        y[0,t-1] = im_np[:,p*int(256/npatch):(p+1)*int(256/npatch)]
                        m[t-1] = mask[:,p*int(256/npatch):(p+1)*int(256/npatch)]
                        
                        if(samplingMode == 'nolattice'):
                            pass
                        else:
                            offset = (offset+1)%4
                    #self.xList.append(x)
                    self.yList.append(y)
                    self.mList.append(m)

    def __getitem__(self, index):
        img = self.mode
        label = self.yList[index]
        mask = self.mList[index]
        
        return img, label, mask

    def __len__(self):
        return len(self.yList)
    

class dataset_1in1_noImg(data.Dataset):
    def __init__(self, iStart = 1,iEnd = 31, mode = 'abs', samplingMode = 'default', reduceMode = ""):
        if(samplingMode == 'fakeRandom'):
            mDic = sio.loadmat(fakeRandomPath)
            miList = mDic['RAll']
        elif(samplingMode == 'rand15'):
            mDic = sio.loadmat(fakeRandomPath_15)
            miList = mDic['RAll']
        self.yList = []
        self.mList = []
        assert mode in ['complex','abs'], "real mode is abandoned"
        self.mode = mode
        offset = 0
        index = 0
        if(reduceMode == "reduce"):
            tList = [1,5,15,20]
        else:
            tList = range(1,21)
        if('nolattice_offset' in samplingMode):
            offset = int(samplingMode[-1])
        for i in range(iStart,iEnd): 
            for z in range(0,5): #0,4
                for t in tList: #1,20
                    #index = (i-iStart)*100+z*20+t-1
                    filename = "data/cardiac_ktz/mr_heart_p%02dt%02dz%d.png" % (i,t,z)
                    im = Image.open(filename)
                    im_np = np.array(im).astype(np.float32)/255.
                    
                    if(samplingMode == 'fakeRandom' or ('rand' in samplingMode)):
                        randI = random.randrange(miList.shape[1])
                        mi = miList[:,randI]
                        mask = subsampling_mask(im_np, 0, 'fakeRandom', mi)
                    else:
                        mask = subsampling_mask(im_np, offset)
                    m = np.fft.ifftshift(mask)
                    
                    y = np.zeros((1,256,256))
                    if(mode == 'abs'):
                        #x = np.zeros((1,256,256))
                        y = np.zeros((1,256,256))
                        #m = mask
                    elif(mode == 'complex'):
                        #x = np.zeros((2,256,256))
                        y = np.zeros((2,256,256))
                    else:
                        assert False,"real mode is abandoned"
                    y[0] = im_np
                    
                    #self.xList.append(x)
                    self.yList.append(y)
                    self.mList.append(m)
                    
                    if(samplingMode == 'nolattice'):
                        pass
                    else:
                        offset = (offset+1)%4

                    index += 1

    def __getitem__(self, index):
        i = index
        img = self.mode
        #label = self.yList[i]
        # if(self.mode == 'complex'):
        #     label = np.zeros((2,256,256))
        #     label[0] = self.yList[i]
        # else:
        label = self.yList[i]
        mask = self.mList[i]
        
        return img, label, mask

    def __len__(self):
        return len(self.yList)