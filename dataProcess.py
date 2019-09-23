import numpy as np
from PIL import Image
import torch.utils.data as data
import scipy.io as sio
import random
import os
import h5py

from util.imageUtil import *

fakeRandomPath = 'mask/mask_r30k_29.mat'
fakeRandomPath_15 = 'mask/mask_r10k_15.mat'
fakeRandomPath_10 = 'mask/mask_r4k_10.mat'
fakeRandomPath_5 = 'mask/mask_r4k_5.mat'

fakeRandomPath_FastMRI_25 = 'mask/mask320_r10k_25.mat'
pathFastMRI_train = 'data/FastMRI/singlecoil_train/'
pathFastMRI_eval = 'data/FastMRI/singlecoil_val/'


def generateDatasetName(configs):
    datasetName = ""
    for index in range(len(configs)):
        if(configs[index]==""):
            continue
        datasetName += configs[index]
        datasetName += '_'

    return datasetName[:-1]

def getDataloader(dataType = '1in1',mode = 'train', batchSize = 1, crossValid = 0):
    pathDirData = 'data/cardiac_ktz/'
    if('FastMRI' in dataType):
        dataset = "FastMRI"
    else:
        dataset = "cardiac"
    if(crossValid == 0):
        crossValidCode = ''
        if(mode == 'train'):
            shuffleFlag = True
            r = list(range(1,31))
        else:
            shuffleFlag = False
            r = list(range(31,34))
    else:
        assert crossValid in range(1, 11), "crossValid out of range"
        crossValidCode = 'crossValid'+str(crossValid)
        if(mode == 'train'):
            shuffleFlag = True
            r = list(range(1, (crossValid - 1) * 3 + 1))
            r.extend(list(range(crossValid * 3 + 1, 34)))
        else:
            shuffleFlag = False
            r = list(range((crossValid - 1) * 3 + 1, crossValid * 3 + 1))
    if('reduce' in dataType):
        reduceMode = "reduce"
    else:
        reduceMode = ""
    #=============================
    if('1in1' in dataType):
        mainType = '1in1'
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
    if('static' in dataType):
        staticSampling = True
        staticPrefix = "static"
    else:
        staticSampling = False
        staticPrefix = ""
    if('random' in dataType):
        samplingMode = 'random'
    elif(('r15' in dataType) or ('rand15' in dataType)):
        samplingMode = 'rand15'
    elif(('r10' in dataType) or ('rand10' in dataType)):
        samplingMode = 'rand10'
    elif(('r5' in dataType) or ('rand5' in dataType)):
        samplingMode = 'rand5'
    elif('fakeRandom' in dataType):
        samplingMode = 'fakeRandom'
    elif('nolattice' in dataType):
        samplingMode = 'nolattice'
    elif('lattice8' in dataType):
        samplingMode = 'lattice8'
    else:
        samplingMode = 'lattice'
        #assert False,"DataType ERROR: No samplingMode Include"
    #-----------------------
    datasetName = generateDatasetName([dataset,mainType,dataMode,staticPrefix+samplingMode,mode,reduceMode,crossValidCode])
    print('#Generating dataset:'+datasetName)
    #=============================
    if(dataset == 'FastMRI'):
        dataset = FastMRI_1in1_noImg(shuffleFlag, dataMode, samplingMode, reduceMode, staticSampling)
    elif(mainType == '1in1'):
        dataset = dataset_1in1_noImg(r, dataMode, samplingMode, reduceMode, staticSampling)
    else:
        assert False,"wrong dataset type"
        
    #flag = (mode == 'train')
    data_loader = data.DataLoader(dataset, batch_size=batchSize, shuffle=shuffleFlag)
    datasize = len(dataset)
    return data_loader,datasize

class dataset_1in1_noImg(data.Dataset):
    def __init__(self, iRange = range(1,31), mode = 'abs', samplingMode = 'default', reduceMode = "", staticRandom = False):
        if(samplingMode == 'random'):
            mDic = sio.loadmat(fakeRandomPath)
            miList = mDic['RAll']
        elif(samplingMode == 'rand15'):
            mDic = sio.loadmat(fakeRandomPath_15)
            miList = mDic['RAll']
        elif(samplingMode == 'rand10'):
            mDic = sio.loadmat(fakeRandomPath_10)
            miList = mDic['RAll']
        elif(samplingMode == 'rand5'):
            mDic = sio.loadmat(fakeRandomPath_5)
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
        for i in iRange:
            for z in range(0,5): #0,4
                for t in tList: #1,20
                    filename = "data/cardiac_ktz/mr_heart_p%02dt%02dz%d.png" % (i,t,z)
                    im = Image.open(filename)
                    im_np = np.array(im).astype(np.float32)/255.
                    
                    if(samplingMode == 'fakeRandom' or ('rand' in samplingMode)):
                        if(staticRandom):
                            randI = index
                        else:
                            randI = random.randrange(miList.shape[1])
                        mi = miList[:,randI]
                        mask = subsampling_mask(im_np, 0, 'fakeRandom', mi)
                    elif(samplingMode == 'lattice8'):
                        mask = subsampling_mask(im_np, offset, 'lattice8')
                    else:
                        mask = subsampling_mask(im_np, offset)
                    m = np.fft.ifftshift(mask)
                    
                    y = np.zeros((1,256,256))
                    if(mode == 'abs'):
                        y = np.zeros((1,256,256))
                    elif(mode == 'complex'):
                        y = np.zeros((2,256,256))
                    else:
                        assert False,"real mode is abandoned"
                    y[0] = im_np

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
        label = self.yList[i]
        mask = self.mList[i]
        
        return img, label, mask

    def __len__(self):
        return len(self.yList)

class FastMRI_1in1_noImg(data.Dataset):
    def __init__(self, isTrain = True, mode = 'abs', samplingMode = 'default', reduceMode = "", staticRandom = False):
        if(samplingMode == 'random'):
            mDic = sio.loadmat(fakeRandomPath_FastMRI_25)
            miList = mDic['RAll']
        else:
            assert False, "FastMRI accept fakeRandom only"
        if(reduceMode == "reduce"):
            isReduce = True
            pathDir = pathFastMRI_eval
            if(isTrain):
                ir_start=0
                ir_end=180
            else:
                ir_start=180
                ir_end=199
        else:
            isReduce = False
            if(isTrain):
                pathDir = pathFastMRI_train
            else:
                pathDir = pathFastMRI_eval
        if(os.path.exists(pathDir)):
            listF = os.listdir(pathDir)
        else:
            assert False, "no such path:" + pathDir

        self.yList = []
        self.mList = []
        assert mode in ['complex','abs'], "real mode is abandoned"
        self.mode = mode
        offset = 0
        index = 0
        SIZE = 320
        
        if('nolattice_offset' in samplingMode):
            offset = int(samplingMode[-1])
        for filename in listF:
            if(isReduce):
                if(index<ir_start or index>=ir_end):
                    index += 1
                    continue
            f = h5py.File(pathDir + filename, 'r')
            esc = np.array(f['reconstruction_esc'])
            for j in range(esc.shape[0]):
                im_np = esc[j].astype(np.float32)
                im_np = im_normalize(im_np)
                im_np = im_np.clip(-6,6)
                if(samplingMode == 'fakeRandom' or ('rand' in samplingMode)):
                    if(staticRandom):
                        randI = index
                    else:
                        randI = random.randrange(miList.shape[1])
                    mi = miList[:,randI]
                    mask = subsampling_mask(im_np, 0, 'fakeRandom', mi)
                else:
                    assert False, "FastMRI accept fakeRandom only"
                    mask = subsampling_mask(im_np, offset)
                m = np.fft.ifftshift(mask)
                
                y = np.zeros((1,SIZE,SIZE))
                if(mode == 'abs'):
                    y = np.zeros((1,SIZE,SIZE))
                elif(mode == 'complex'):
                    y = np.zeros((2,SIZE,SIZE))
                else:
                    assert False,"real mode is abandoned"
                y[0] = im_np
                
                self.yList.append(y)
                self.mList.append(m)
                
                if(samplingMode == 'nolattice'):
                    pass
                else:
                    offset = (offset+1)%4

                index += 1
            f.close()


    def __getitem__(self, index):
        i = index
        img = self.mode
        label = self.yList[i]
        mask = self.mList[i]
        
        return img, label, mask

    def __len__(self):
        return len(self.yList)