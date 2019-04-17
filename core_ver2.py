import matplotlib.pyplot as plt
#%matplotlib inline
#import torchvision
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import os
import numpy as np
from PIL import Image
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
import configparser

from util import imshow, img2f, f2img, kspace_subsampling, Recoder, paramNumber, kspace_subsampling_pytorch, imgFromSubF_pytorch
from network import getNet, getNet_2net, getLoss, getOptimizer
from dataProcess import getDataloader

class core():
    def __init__(self,configPath, isEvaluate = False):
        self.config = configparser.ConfigParser()
        self.config.read(configPath)

        self.batchSize = int(self.config['train']['batchSize'])
        self.epoch = int(self.config['train']['epoch'])
        self.LR = float(self.config['train']['learningRate'])
        
        self.saveEpoch = int(self.config['log']['saveEpoch'])
        self.maxSaved = int(self.config['log']['maxSaved'])

        self.useCuda = self.config.getboolean('general','useCuda')
        self.needParallel = self.config.getboolean('general','needParallel')
        if(self.useCuda):
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.mode = self.config['general']['mode']
        print("#Result Path:"+self.config['general']['path'])
        print("#Create network:"+self.config['general']['netType'])
        print("#Mode:"+self.mode)
        
        if(self.mode=='2net'): 
            self.net1 = getNet_2net(self.config['general']['netType'], 0).type(self.dtype)
            self.net2 = getNet_2net(self.config['general']['netType'], 1).type(self.dtype)
        else:
            self.net = getNet(self.config['general']['netType']).type(self.dtype)
        if(self.needParallel):
            listDeviceStr = self.config['general']['device']
            listDevice = []
            for i in range(len(listDeviceStr)):
                if(listDeviceStr[i] == '1'):
                    listDevice.append(i)
            assert len(listDevice) > 0,"No device is selected"
            print('parallel device:'+str(listDevice))
            if(self.mode=='2net'): 
                self.net1 = nn.DataParallel(self.net1, device_ids = listDevice).type(self.dtype)
                self.net2 = nn.DataParallel(self.net2, device_ids = listDevice).type(self.dtype)
            else:
                self.net = nn.DataParallel(self.net, device_ids = listDevice).type(self.dtype)
        if(self.mode=='2net'): 
            self.net = nn.ModuleList([self.net1,self.net2])
        self.lossForward = getLoss(self.config['general']['lossType']).type(self.dtype)
        paramNum = paramNumber(self.net)
        print("#Number of network parameters:%d"%paramNum)

        if(not isEvaluate):
            self.trainloader,self.trainsetSize = getDataloader(self.config['general']['dataType'],'train',self.batchSize)
        self.testloader,self.testsetSize = getDataloader(self.config['general']['dataType'],'test',1)

        self.record = Recoder(self.config['general']['path'],self.saveEpoch*self.maxSaved)
        
        if(not isEvaluate):
            self.config.write(open(self.record.rootPath+"/config.ini","w"))
            self.record.logNet(self.net)
            self.record.log("### Number of network parameters:%d"%paramNum)

        #self.optimizer = getOptimizer(self.net.parameters(), self.config['train']['optimizer'], self.LR)
        if(self.config['train']['optimizer'] == 'Adam_wd'):
            self.weightDecay = float(self.config['train']['weightDecay'])
        else:
            self.weightDecay = 0
        print('#Optimizer: '+self.config['train']['optimizer']+' LR = %.2e weightDecay = %.2e'%(self.LR,self.weightDecay))
        self.ckp_epoch = 0
        
    def train(self, epoch = 0, listTarget = [0,1], forwardStage = 1, need_evaluate = True):
        if(epoch == 0):
            epoch = self.epoch
        if(self.mode == '2net'):
            ltmp = []
            for i in listTarget:
                ltmp.append(self.net[i])
            self.optimizer = getOptimizer(nn.ModuleList(ltmp).parameters(), self.config['train']['optimizer'], self.LR, self.weightDecay)
            msg = "start training: listTarget "+str(listTarget)+", stage %d, epoch = %d"%(forwardStage,epoch)
        else:
            self.optimizer = getOptimizer(self.net.parameters(), self.config['train']['optimizer'], self.LR, self.weightDecay)
            msg = "start training: epoch = %d"%(epoch)
        self.record.log(msg)
        print(msg)
        for j in range(1,epoch + 1):
            self.net.train()
            i = 0
            total_loss = 0
            for mode, label, mask in self.trainloader:
                self.optimizer.zero_grad()
                #netInput = Variable(img).type(self.dtype)
                netLabel = Variable(label).type(self.dtype)
                mask_var = Variable(mask).type(self.dtype)
                subF = kspace_subsampling_pytorch(netLabel,mask_var)
                complexFlag = (mode[0] == 'complex')
                netInput = imgFromSubF_pytorch(subF,complexFlag)

                if(self.mode == 'inNetDC'):
                    netOutput = self.net(netInput, subF, mask_var)
                elif(self.mode == 'inNetDC_2out'):
                    netOutput,netBeforeCor = self.net(netInput, subF, mask_var)
                elif(self.mode == '2net'):
                    netOutput = self.net[0](netInput)
                    if(forwardStage>1):
                        #netInput_curFrame = netInput[:,0:netLabel.shape[1]]
                        #netInput_curFrame = subF
                        netOutput = self.net[1](netOutput,subF,mask_var)
                else:
                    netOutput = self.net(netInput)
                
                loss = self.lossForward(netOutput,netLabel)
                loss.backward()
                total_loss = total_loss + loss.item()*label.shape[0]

                self.optimizer.step()
                i += label.shape[0]
                print ('Epoch %05d [%04d/%04d] loss %.8f' % (j+self.ckp_epoch,i,self.trainsetSize,loss.item()), '\r', end='')
            self.record.log_train(j+self.ckp_epoch,total_loss/i)

            if j % self.saveEpoch == 0:
                print ('Epoch %05d [%04d/%04d] loss %.8f SAVED' % (j+self.ckp_epoch,i,self.trainsetSize,total_loss/self.trainsetSize))
                if need_evaluate:
                    l,p1,p2,s1,s2 = self.validation()
                    self.record.log_valid(j+self.ckp_epoch,l,p1,p2,s1,s2)
                    self.record.log("Evaluate psnr(before|after) =  %.2f|%.2f ssim = %.4f|%.4f"%(p1,p2,s1,s2))
                self.record.write_to_file(self.net.state_dict(), False)
        self.ckp_epoch += epoch
        self.record.write_to_file(self.net.state_dict(), True)
    
    def testValue(self,mode,label,mask):
        result = self.test(mode,label,mask)
        
        return result['loss'],result['psnr1'],result['psnr2'],result['ssim1'],result['ssim2']
    
    def test(self,mode,label,mask):
        y = label.numpy() 
        #netInput = Variable(img).type(self.dtype)
        netLabel = Variable(label).type(self.dtype)
        if(self.mode == 'normal'):
            assert False, 'normal mode is abandoned'
            # netInput = Variable(img).type(self.dtype)
            # x = img.numpy()
            # mask = mask.numpy()[0]
        else:
            mask_var = Variable(mask).type(self.dtype)
            subF = kspace_subsampling_pytorch(netLabel,mask_var)
            complexFlag = (mode[0] == 'complex')
            netInput = imgFromSubF_pytorch(subF,complexFlag)
            
        y=y[0]
            
        if(self.mode == 'inNetDC'): 
            netOutput = self.net(netInput, subF, mask_var)
            loss = self.lossForward(netOutput,netLabel)
            netOutput_np = netOutput.cpu().data.numpy()
            img1 = netOutput_np[0,0:1].astype('float64')
            if(netOutput_np.shape[1]==2):
                netOutput_np = abs(netOutput_np[:,0:1]+netOutput_np[:,1:2]*1j)
            img2 = netOutput_np[0].astype('float64')
            y2 = y[0:1]
        elif(self.mode == 'inNetDC_2out'): 
            netOutput,netBeforeCor = self.net(netInput, subF, mask_var)
            loss = self.lossForward(netOutput,netLabel)
            netOutput_np = netOutput.cpu().data.numpy()
            netBeforeCor_np = netBeforeCor.cpu().data.numpy()
            if(netOutput_np.shape[1]==2):
                netOutput_np = abs(netOutput_np[:,0:1]+netOutput_np[:,1:2]*1j)
            if(netBeforeCor_np.shape[1]==2):
                netBeforeCor_np = abs(netBeforeCor_np[:,0:1]+netBeforeCor_np[:,1:2]*1j)
            img1 = netBeforeCor_np[0].astype('float64')
            img2 = netOutput_np[0].astype('float64')
            y2 = y[0:1]
        elif(self.mode == '2net'):
            netOutput = self.net1(netInput)
            # if(netInput.shape[1]==3):
            #     netInput_curFrame = netInput[:,1:2]
            # else:
            #     netInput_curFrame = netInput
            #netInput_curFrame = subF
            netCorrect = self.net2(netOutput,subF,mask_var)
            loss = self.lossForward(netCorrect,netLabel)
            netOutput_np = netOutput.cpu().data.numpy()
            netCorrect_np = netCorrect.cpu().data.numpy()
            if(netOutput_np.shape[1]==2):
                netOutput_np = abs(netOutput_np[:,0:1]+netOutput_np[:,1:2]*1j)
            if(netCorrect_np.shape[1]==2):
                netCorrect_np = abs(netCorrect_np[:,0:1]+netCorrect_np[:,1:2]*1j)
            img1 = netOutput_np[0].astype('float64')
            img2 = netCorrect_np[0].astype('float64')
            y2 = y[0:1]
        else:
            netOutput = self.net(netInput)
            loss = self.lossForward(netOutput,netLabel)
            netOutput_np = netOutput.cpu().data.numpy()[0]
            img1 = netOutput_np.astype('float64')

            y_f = img2f(netOutput_np)
            if(x.shape[1]==3):
                x_f = img2f(x[:,1,:,:])
            else:
                x_f = img2f(x[:,0])
            y_f = y_f - y_f * mask
            y_f = y_f + x_f * mask

            img2 = np.abs(f2img(y_f)).astype('float64')
            y2 = y
            
        img1 = np.clip(img1,0,1)
        img2 = np.clip(img2,0,1)
        
        psnrBefore = psnr(y2,img1)
        psnrAfter = psnr(y2,img2)
        
        ssimBefore = ssim(y2[0],img1[0])
        ssimAfter = ssim(y2[0],img2[0])
        
        return {"loss":loss.item(),"psnr1":psnrBefore,"psnr2":psnrAfter,"ssim1":ssimBefore,"ssim2":ssimAfter,"result1":img1,"result2":img2,'label':y2}
    
    def validation(self,returnList = False):
        self.net.eval()
        i = 0
        totalLoss = 0
        lpsnr1 = []
        lpsnr2 = []
        lssim1 = []
        lssim2 = []
        psnr1 = 0
        psnr2 = 0
        ssim1 = 0
        ssim2 = 0
        # self.dataloaderCheck()
        for mode, label, mask in self.testloader:
            loss0,psnrA,psnrB,ssimA,ssimB = self.testValue(mode,label,mask)
            
            totalLoss += loss0
            psnr1 += psnrA
            psnr2 += psnrB
            ssim1 += ssimA
            ssim2 += ssimB
            lpsnr1.append(psnrA)
            lpsnr2.append(psnrB)
            lssim1.append(ssimA)
            lssim2.append(ssimB)
            i+=1
            print ('Evaluating %04d psnr(before|after) =  %.2f|%.2f ssim = %.4f|%.4f' % (i, psnrA, psnrB, ssimA, ssimB), '\r', end='')
        print ('Evaluating %04d psnr(before|after) =  %.2f|%.2f ssim = %.4f|%.4f Done' % (i, psnr1/i, psnr2/i, ssim1/i, ssim2/i))
        if(returnList):
            return lpsnr1,lpsnr2,lssim1,lssim2
        return totalLoss/i, psnr1/i, psnr2/i, ssim1/i, ssim2/i
    
    def loadCkpt(self, expectedEpoch, isChecked=False):
        self.record.load_from_file(self.net, expectedEpoch, isChecked, False)
        self.ckp_epoch = expectedEpoch