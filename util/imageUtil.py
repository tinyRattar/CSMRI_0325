import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

def imshow(img,mode='pil',vmax=0,overlap=False):
    #assert False, no more imshow
    if(mode == 'cv'):
        b,g,r = cv2.split(img)
        im_np = cv2.merge([r,g,b])
        plt.imshow(im_np)
    elif(mode == 'pil'):
        if(vmax!=0):
            plt.imshow(img,vmax=vmax)
        else:
            plt.imshow(img)
    elif(mode == 'g'):
        plt.imshow(img,'gray')
    elif(mode == 'b'):
        plt.imshow(img,'binary')
    elif(mode == 'c'):
        if('complex' in str(img.dtype)):
            showImg = img
        else:
            showImg = fc2c(img)
        plt.imshow(abs(showImg),'gray')
    elif(mode == 'k'):
        draw_kspace_img(img)
    else:
        assert False,"wrong mode"
    
    plt.axis('off')
    if(~overlap):
        plt.show()

def fc2c(img):
    '''
    from fake_complex np.array, shape = [2,...] to complex np.array
    '''
    assert img.shape[0]==2,"the first dimension of img should be 2"
    outImg = img[0]+img[1]*1j

    return outImg


def draw_kspace_img(img):
    s1 = np.log(np.abs(img))
    plt.imshow(s1,'gray')
    
def img2f(img):
    f = np.fft.fft2(img, norm='ortho')
    f = np.fft.fftshift(f)
    
    return f

def f2img(f):
    f = np.fft.ifftshift(f)
    img = np.fft.ifft2(f, norm='ortho')
    
    return img

def kspace_subsampling(srcImg,offset = 0,mode="default",mi=None):
    mask = np.zeros_like(srcImg)
    if(mode == "default"):
        assert offset<4 , "offset out of range"
        #mask = np.zeros((256,256))
        mask[offset::4,:] = 1
        mask[122:134,:] = 1
    elif(mode == "lattice8"):
        assert offset<8 , "offset out of range"
        #mask = np.zeros((256,256))
        mask[offset::8,:] = 1
        mask[125:131,:] = 1
    elif(mode == "fakeRandom"):
        mask[mi==1,:] = 1
    else:
        assert False, "wrong subsampling mode"

    srcF = np.fft.fft2(srcImg, norm='ortho')
    srcF = np.fft.fftshift(srcF)
    tarF = srcF*mask
    
    return tarF,mask

def subsampling_mask(srcImg,offset=0, mode = "default", mi = None):
    mask = np.zeros_like(srcImg)
    if(mode == "default"):
        assert offset<4 , "offset out of range"
        #mask = np.zeros((256,256))
        mask[offset::4,:] = 1
        mask[122:134,:] = 1
    elif(mode == "lattice8"):
        assert offset<8 , "offset out of range"
        #mask = np.zeros((256,256))
        mask[offset::8,:] = 1
        mask[125:131,:] = 1
    elif(mode == "fakeRandom"):
        mask[mi==1,:] = 1
    else:
        assert False, "wrong subsampling mode"
    
    return mask

def addZoomIn(img, x0 = 87, y0 = 135,offsetX=32,offsetY=0,scale=3,border = 0):
    if(len(img.shape)==3):
        im1 = np.zeros_like(img)
        im1 = img.copy()
    else:
        im1 = np.zeros((img.shape[0],img.shape[1],3))
        for i in range(3):
            im1[:,:,i] = img
    if(offsetY==0):
        offsetY = offsetX
    #scale = 3
    imzoomin = im1[y0:y0+offsetY,x0:x0+offsetX]
    imzoomin = cv2.resize(imzoomin,((offsetY*scale,offsetX*scale)))
    cv2.rectangle(im1,(x0,y0),(x0+offsetX,y0+offsetY),(255,0,0),1)
    im1[256-offsetY*scale:,256-offsetX*scale:] = imzoomin
    if(border>0):
        im1[-offsetX*scale-border:-offsetX*scale,256-offsetX*scale-border:] = (0,0,0)
        im1[256-offsetY*scale-border:,-offsetX*scale-border:-offsetX*scale] = (0,0,0)
    
    return im1

def saveIm(img,path):
    b,g,r = cv2.split(img) 
    img2 = cv2.merge([r,g,b])
    cv2.imwrite(path,img2)
    
def rgb2bgr(img):
    b,g,r = cv2.split(img) 
    img2 = cv2.merge([r,g,b])
    
    return img2

def kspace_subsampling_pytorch(srcImg,mask):
    '''
    return subF has shape[...,2], without permute
    '''
    y = srcImg
    if(len(y.shape)==4):
        if(y.shape[1]==1):
            emptyImag = torch.zeros_like(y)
            xGT_c = torch.cat([y,emptyImag],1).permute(0,2,3,1)
        else:
            xGT_c = y.permute(0,2,3,1)
        mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],1)
    elif(len(y.shape)==5):
        if(y.shape[1]==1):
            emptyImag = torch.zeros_like(y)
            xGT_c = torch.cat([y,emptyImag],1).permute(0,2,3,4,1)
        else:
            xGT_c = y.permute(0,2,3,4,1)
        mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],mask.shape[3],1)
    else:
        assert False, "srcImg shape length has to be 4(2d) or 5(3d)"
    
    xGT_f = torch.fft(xGT_c,2, normalized=True)
    subF = xGT_f * mask

    # if(len(y.shape)==4):
    #         subF = subF.permute(0,3,1,2)
    #     else:
    #         subF = subF.permute(0,4,1,2,3)

    return subF

def imgFromSubF_pytorch(subF,returnComplex=False):
    subIm = torch.ifft(subF,2, normalized=True)
    if(len(subIm.shape)==4):
        subIm = subIm.permute(0,3,1,2)
    else:
        subIm = subIm.permute(0,4,1,2,3)

    if(returnComplex):
        return subIm
    else:
        subIm = torch.sqrt(subIm[:,0:1]*subIm[:,0:1]+subIm[:,1:2]*subIm[:,1:2])
        return subIm

def gaussianFilter(H,W,radius,highpass = True, needShift = True):
    fmask = np.zeros((H,W))

    for i in range(H):
        for j in range(W):
            D = np.sqrt((i-H/2)*(i-H/2)+(j-W/2)*(j-W/2));
            if(highpass):
                fmask[i,j] = 1 - np.exp(-D*D/2/(radius*radius))
            else:
                fmask[i,j] = np.exp(-D*D/2/(radius*radius))

    if(needShift):
        fmask = np.fft.fftshift(fmask)

    return fmask


