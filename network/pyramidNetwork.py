import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .networkUtil import *

#activLayer = nn.LeakyReLU()
activLayer = nn.ReLU()

class convBlock(nn.Module):
    def __init__(self, inChannel = 2, outChannel = 2, features=32, kernelSize=3, dilateMulti = 1,layer = 5):
        super(convBlock, self).__init__()
        self.layer = layer
        
        templayerList = []
        dilate = 1
        if(self.layer == 1):
        	features = inChannel
        for i in range(0, self.layer - 1):
            if(i == 0):
                tempConv = nn.Conv2d(inChannel, features, kernelSize, padding = int(dilate * (kernelSize - 1) / 2), dilation = dilate)
            else:
                tempConv = nn.Conv2d(features, features, kernelSize, padding = int(dilate * (kernelSize - 1) / 2), dilation = dilate)
            templayerList.append(tempConv)
            dilate = dilate * dilateMulti
        tempConv = nn.Conv2d(features, outChannel, kernelSize, padding = int(dilate * (kernelSize - 1) / 2), dilation = dilate)
        templayerList.append(tempConv)
        self.layerList = nn.ModuleList(templayerList)
        self.relu = activLayer
    
    def forward(self,x1):
        x = x1
        for i in range(0, self.layer - 1):
            x = self.layerList[i](x)
            x = self.relu(x)
        y = self.layerList[self.layer - 1](x)
        
        return y

class denseBlock_tr(nn.Module):
    def __init__(self, inChannel=16, outChannel=16, growthRate=16, kernelSize=3, dilationLayer = False, layer = 3, activ = 'ReLU', useOri = True, transition = 0.25):
        super(denseBlock_tr, self).__init__()
        dilate = 1
        if(dilationLayer):
            dilateMulti = 2
        else:
            dilateMulti = 1
        pad = int((kernelSize-1)/2)
        self.denselayer = layer
        templayerList = []
        for i in range(0, self.denselayer):
            if(useOri):
                tempLayer = denseBlockLayer_origin(inChannel+growthRate*i, growthRate, kernelSize, inChannel, dilate, activ)
            else:
                tempLayer = denseBlockLayer(inChannel+growthRate*i, growthRate, kernelSize, False, dilate, activ)
            dilate = dilate * dilateMulti
            templayerList.append(tempLayer)
        self.layerList = nn.ModuleList(templayerList)
        self.transitionLayer = transitionLayer(inChannel+growthRate*layer, transition, activ = activ)
            
    def forward(self,x):
        for i in range(0, self.denselayer):
            tempY = self.layerList[i](x)
            x = torch.cat((x, tempY), 1)
        y = self.transitionLayer(x)

        return y

def boxDownsampleInKspace(x):
	shape = x.shape
	if(shape[-1] ==2):
		h = int(shape[-3]/4)
		w = int(shape[-2]/4)
		assert shape[-2] % 4 ==0, "input shape error in k-space downsampler"
		x_ds = x[...,:2*h,:2*w,:]
		x_ds[...,h:2*h,:w,:] = x[...,3*h:4*h,:w,:]
		x_ds[...,:h,w:2*w,:] = x[...,:h,3*w:4*w,:]
		x_ds[...,h:2*h,w:2*w,:] = x[...,3*h:4*h,3*w:4*w,:]
	else:
		h = int(shape[-2]/4)
		w = int(shape[-1]/4)
		assert shape[-1] % 4 ==0, "input shape error in k-space downsampler"
		x_ds = x[...,:2*h,:2*w]
		x_ds[...,h:2*h,:w] = x[...,3*h:4*h,:w]
		x_ds[...,:h,w:2*w] = x[...,:h,3*w:4*w]
		x_ds[...,h:2*h,w:2*w] = x[...,3*h:4*h,3*w:4*w]

	return x_ds

def ifft_torch(k):
	xout = torch.ifft(k,2, normalized=True)
	if(len(xout.shape)==4):
		xout = xout.permute(0,3,1,2)
	else:
		xout = xout.permute(0,4,1,2,3)

	return xout

class vanillaPyramidNetwork(nn.Module):
	def __init__(self, fNum=32, isDilate = False, dcTrick = 0):
		super(vanillaPyramidNetwork, self).__init__()
		InterBlock = convBlock
		Encoder = convBlock
		Decoder = convBlock
		Downsampler = nn.MaxPool2d
		Upsampler = nn.ConvTranspose2d
		UpsamplerF = nn.ConvTranspose2d
		DataConsistency = dataConsistencyLayer_static
		# Block 0
		self.b0_en = Encoder(2, fNum, fNum, layer = 1)
		self.b0_inter = InterBlock(fNum, fNum, fNum, layer = 3)

		# Block 1
		self.b1_ds = Downsampler(kernel_size = 2)
		self.b1_inter = InterBlock(fNum, fNum, fNum, layer = 3)

		# Block 2
		self.b2_ds = Downsampler(kernel_size = 2)
		self.b2_inter = InterBlock(fNum, fNum, fNum, layer = 3)
		# --us
		self.b2_us = Upsampler(fNum, fNum, 2, 2)
		# --generator
		self.b2_de = Decoder(fNum, 2, fNum, layer = 1)
		self.b2_dc = DataConsistency(trick = dcTrick)
		self.b2_usf = UpsamplerF(2, 2, 2, 2)

		# Block 3
		self.b3_inter = InterBlock(fNum, fNum, fNum, layer = 3)
		# --us
		self.b3_us = Upsampler(fNum, fNum, 2, 2)
		# --generator
		self.b3_de = Decoder(fNum, 2, fNum, layer = 1)
		self.b3_dc = DataConsistency(trick = dcTrick)
		self.b3_usf = UpsamplerF(2, 2, 2, 2)

		# Block 4
		self.b4_inter = InterBlock(fNum, fNum, fNum, layer = 3)
		# --generator
		self.b4_de = Decoder(fNum, 2, fNum, layer = 1)
		self.b4_dc = DataConsistency(trick = dcTrick)

		self.activ = activLayer

	def forward(self, x, y, mask):
		y_ds2x = boxDownsampleInKspace(y)
		y_ds4x = boxDownsampleInKspace(y_ds2x)
		yIm_ds4x = ifft_torch(y_ds4x)

		mask_ds2x = boxDownsampleInKspace(mask)
		mask_ds4x = boxDownsampleInKspace(mask_ds2x)

		f0 = self.b0_en(x)
		f0 = self.activ(f0)
		f0 = self.b0_inter(f0)

		f0 = self.activ(f0)

		f1 = self.b1_ds(f0)
		f1 = self.b1_inter(f1)
		f1 = self.activ(f1)

		f2 = self.b2_ds(f1)
		f2 = self.b2_inter(f2)
		f2 = self.activ(f2)

		xg2 = self.b2_de(f2)
		xg2 += yIm_ds4x
		xg2 = self.b2_dc(xg2, y_ds4x, mask_ds4x)
		xg2u = self.b2_usf(xg2)

		f2u = self.b2_us(f2)
		f2u = self.activ(f2u)
		f3 = self.b3_inter(f2u)
		f3 = self.activ(f3)

		xg3 = self.b3_de(f3)
		xg3 += xg2u
		xg3 = self.b3_dc(xg3, y_ds2x, mask_ds2x)
		xg3u = self.b3_usf(xg3)

		f3u = self.b3_us(f3)
		f3u = self.activ(f3u)
		f4 = self.b4_inter(f3u)
		f4 = self.activ(f4)

		xg4 = self.b4_de(f4)
		xg4 += xg3u
		xg4 = self.b4_dc(xg4, y , mask)

		return xg4

class conv(nn.Module):
	def __init__(self, inChannel = 2, outChannel = 32):
		super(conv, self).__init__()
		self.conv = nn.Conv2d(inChannel,outChannel,3,padding=1)
		self.relu = activLayer

	def forward(self,x):
		f = self.conv(x)
		y = self.relu(f)

		return y


# just for debug
class vanillaPyramidNetwork_debug(nn.Module):
    def __init__(self, fNum=32, isDilate = False, dcTrick = 0):
        super(vanillaPyramidNetwork_debug, self).__init__()
        # Block 0
        self.block0 = nn.Sequential()
        self.block0.add_module("conv_0", conv(2,32))
        self.block0.add_module("conv_11", conv(32,32))
        self.block0.add_module("conv_12", conv(32,32))
        self.block0.add_module("conv_13", conv(32,32))
        self.block0.add_module("mp1", nn.MaxPool2d(kernel_size = 2))
        self.block0.add_module("conv_21", conv(32,32))
        self.block0.add_module("conv_22", conv(32,32))
        self.block0.add_module("conv_23", conv(32,32))
        self.block0.add_module("mp2", nn.MaxPool2d(kernel_size = 2))
        self.block0.add_module("conv_31", conv(32,32))
        self.block0.add_module("conv_32", conv(32,32))
        self.block0.add_module("conv_33", conv(32,32))
        self.block0.add_module("dc1", nn.ConvTranspose2d(32,32,2,2))
        self.block0.add_module("r1", nn.ReLU())
        self.block0.add_module("conv_41", conv(32,32))
        self.block0.add_module("conv_42", conv(32,32))
        self.block0.add_module("conv_43", conv(32,32))
        self.block0.add_module("dc2", nn.ConvTranspose2d(32,32,2,2))
        self.block0.add_module("r2", nn.ReLU())
        self.block0.add_module("conv_51", conv(32,32))
        self.block0.add_module("conv_52", conv(32,32))
        self.block0.add_module("conv_53", conv(32,32))
        self.block0.add_module("conv_out", nn.Conv2d(32,2,3,padding = 1))

        self.dc = dataConsistencyLayer_static(trick = 0)


    def forward(self, x, y, mask):
        x2 = self.block0(x)
        x2 = x2+x
        x2 = self.dc(x2,y,mask)

        return x2

class vanillaPyramidNetwork2(nn.Module):
	def __init__(self, fNum=16, isDilate = False, dcTrick = 0, debug = 0):
		super(vanillaPyramidNetwork2, self).__init__()
		self.debug = debug
		InterBlock = denseBlock_tr
		Encoder = convBlock
		Decoder = convBlock
		Downsampler = nn.MaxPool2d
		Upsampler = nn.ConvTranspose2d
		UpsamplerF = nn.ConvTranspose2d
		DataConsistency = dataConsistencyLayer_static
		# Block 0
		self.b0_en = Encoder(2, fNum, fNum, layer = 1)
		self.b0_inter = InterBlock(fNum, fNum, fNum, layer = 3)

		# Block 1
		self.b1_ds = Downsampler(kernel_size = 2)
		self.b1_inter = InterBlock(fNum, fNum, fNum, layer = 3)

		# Block 2
		self.b2_ds = Downsampler(kernel_size = 2)
		self.b2_inter = InterBlock(fNum, fNum, fNum, layer = 3)
		# --generator
		self.b2_de = Decoder(fNum, 2, fNum, layer = 1)
		self.b2_dc = DataConsistency(trick = dcTrick)
		self.b2_usf = UpsamplerF(2, 2, 2, 2)

		# Block 3
		self.b3_en = Encoder(2, fNum, fNum, layer = 1)
		self.b3_inter = InterBlock(fNum, fNum, fNum, layer = 3)
		# --generator
		self.b3_de = Decoder(fNum, 2, fNum, layer = 1)
		self.b3_dc = DataConsistency(trick = dcTrick)
		self.b3_usf = UpsamplerF(2, 2, 2, 2)

		# Block 4
		self.b4_en = Encoder(2, fNum, fNum, layer = 1)
		self.b4_inter = InterBlock(fNum, fNum, fNum, layer = 3)
		# --generator
		self.b4_de = Decoder(fNum, 2, fNum, layer = 1)
		self.b4_dc = DataConsistency(trick = dcTrick)

		self.activ = activLayer

	def forward(self, x, y, mask):
		y_ds2x = boxDownsampleInKspace(y)
		y_ds4x = boxDownsampleInKspace(y_ds2x)
		yIm_ds2x = ifft_torch(y_ds2x)
		yIm_ds4x = ifft_torch(y_ds4x)

		mask_ds2x = boxDownsampleInKspace(mask)
		mask_ds4x = boxDownsampleInKspace(mask_ds2x)

		f0 = self.b0_en(x)
		f0 = self.activ(f0)
		f0 = self.b0_inter(f0)
		f0 = self.activ(f0)

		f1 = self.b1_ds(f0)
		f1 = self.b1_inter(f1)
		f1 = self.activ(f1)

		f2 = self.b2_ds(f1)
		f2 = self.b2_inter(f2)
		f2 = self.activ(f2)

		xg2 = self.b2_de(f2)
		#xg2 += yIm_ds4x
		#xg2 = self.b2_dc(xg2, y_ds4x, mask_ds4x)
		xg2u = self.b2_usf(xg2)

		f3 = self.b3_en(xg2u)
		f3 = self.activ(f3)
		f3 = self.b3_inter(f3)
		f3 = self.activ(f3)

		xg3 = self.b3_de(f3)
		#xg3 += yIm_ds2x
		#xg3 = self.b3_dc(xg3, y_ds2x, mask_ds2x)
		xg3u = self.b3_usf(xg3)

		f4 = self.b4_en(xg3u)
		f4 = self.activ(f4)
		f4 = self.b4_inter(f4)
		f4 = self.activ(f4)

		xg4 = self.b4_de(f4)
		xg4 += x
		xg4 = self.b4_dc(xg4, y, mask)

		return xg4


class vanillaPyramidNetwork3(nn.Module):
	def __init__(self, fNum=16, isDilate = False, dcTrick = 0, debug = 0):
		super(vanillaPyramidNetwork3, self).__init__()
		self.debug = debug
		InterBlock = denseBlock_tr
		Encoder = convBlock
		Decoder = convBlock
		Downsampler = nn.MaxPool2d
		Upsampler = nn.ConvTranspose2d
		UpsamplerF = nn.ConvTranspose2d
		DataConsistency = dataConsistencyLayer_static
		# Block 0
		self.b0_en = Encoder(2, fNum, fNum, layer = 1)
		self.b0_inter = InterBlock(fNum, fNum, fNum, layer = 3)

		# Block 1
		self.b1_ds = Downsampler(kernel_size = 2)
		self.b1_inter = InterBlock(fNum, fNum, fNum, layer = 3)

		# Block 2
		self.b2_ds = Downsampler(kernel_size = 2)
		self.b2_inter = InterBlock(fNum, fNum, fNum, layer = 3)
		# --generator
		self.b2_de = Decoder(fNum, 2, fNum, layer = 1)
		self.b2_dc = DataConsistency(trick = dcTrick)
		self.b2_usf = UpsamplerF(2, 2, 2, 2)

		# Block 3
		self.b3_en = Encoder(2, fNum, fNum, layer = 1)
		self.b3_inter = InterBlock(fNum, fNum, fNum, layer = 3)
		# --generator
		self.b3_de = Decoder(fNum, 2, fNum, layer = 1)
		self.b3_dc = DataConsistency(trick = dcTrick)
		self.b3_usf = UpsamplerF(2, 2, 2, 2)

		# Block 4
		self.b4_en = Encoder(2, fNum, fNum, layer = 1)
		self.b4_inter = InterBlock(fNum, fNum, fNum, layer = 3)
		# --generator
		self.b4_de = Decoder(fNum, 2, fNum, layer = 1)
		self.b4_dc = DataConsistency(trick = dcTrick)

		self.activ = activLayer

	def forward(self, x, y, mask):
		y_ds2x = boxDownsampleInKspace(y)
		y_ds4x = boxDownsampleInKspace(y_ds2x)
		yIm_ds2x = ifft_torch(y_ds2x)
		yIm_ds4x = ifft_torch(y_ds4x)

		mask_ds2x = boxDownsampleInKspace(mask)
		mask_ds4x = boxDownsampleInKspace(mask_ds2x)

		f0 = self.b0_en(x)
		f0 = self.activ(f0)
		f0 = self.b0_inter(f0)
		f0 = self.activ(f0)

		f1 = self.b1_ds(f0)
		f1 = self.b1_inter(f1)
		f1 = self.activ(f1)

		# f2 = self.b2_ds(f1)
		# f2 = self.b2_inter(f2)
		# f2 = self.activ(f2)

		# xg2 = self.b2_de(f2)
		# xg2 += yIm_ds4x
		# xg2 = self.b2_dc(xg2, y_ds4x, mask_ds4x)
		# xg2u = self.b2_usf(xg2)

		# f3 = self.b3_en(xg2u)
		# f3 = self.activ(f3)
		# f3 = self.b3_inter(f3)
		# f3 = self.activ(f3)

		xg3 = self.b3_de(f1)
		xg3 += yIm_ds2x
		xg3 = self.b3_dc(xg3, y_ds2x, mask_ds2x)
		xg3u = self.b3_usf(xg3)

		f4 = self.b4_en(xg3u)
		f4 = self.activ(f4)
		f4 = self.b4_inter(f4)
		f4 = self.activ(f4)

		xg4 = self.b4_de(f4)
		xg4 += x
		xg4 = self.b4_dc(xg4, y, mask)

		return xg4