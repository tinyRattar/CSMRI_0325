import datetime
import torch
import scipy.io as sio
import os
import numpy as np

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def saveNet(param, saveDir, epochNum, epochOffset, checkpoint=False):
    tmpDir = saveDir
    tmpPath = tmpDir+"/saved"+"_"+str(epochNum)+".pkl"
    if(checkpoint):
        tmpPath = tmpDir+"/CHECKED_saved"+"_"+str(epochNum)+".pkl"
    oldNum = epochNum - epochOffset
    oldPath = tmpDir+"/saved"+"_"+str(oldNum)+".pkl"
    if(os.path.exists(oldPath)):
        os.remove(oldPath)
    torch.save(param,tmpPath)
    
def loadNet(net, loadDir, epochNum, checkpoint=False):
    tmpDir = loadDir
    tmpPath = tmpDir+"/saved"+"_"+str(epochNum)+".pkl"
    if(checkpoint):
        tmpPath = tmpDir+"/CHECKED_saved"+"_"+str(epochNum)+".pkl"
    net.load_state_dict(torch.load(tmpPath))

class Recoder():
    def __init__(self,path,offsetEpoch):
        self.trained_epoch = 0 
        self.save_offset = offsetEpoch # for delete old weight file
        self.trainRecord = {"epoch":[], "trainLoss":[]}
        self.validRecord = {"epoch":[], "validLoss":[], "PSNR":[], "PSNRafter":[], "SSIM":[], "SSIMafter":[]}
        self.rootPath = path
        self.logPath = self.rootPath+'/log'
        self.weightPath = self.rootPath+'/weight'
        mkdir(self.logPath)
        mkdir(self.weightPath)
    
    def log_train(self,epoch0,loss0):
        self.trainRecord["epoch"].append(epoch0)
        self.trained_epoch = epoch0
        self.trainRecord["trainLoss"].append(loss0)
    
    def log_valid(self,epoch0,loss0,psnr0,psnrafter0,ssim0,ssimafter0):
        self.validRecord["epoch"].append(epoch0)
        self.validRecord["validLoss"].append(loss0)
        self.validRecord["PSNR"].append(psnr0)
        self.validRecord["PSNRafter"].append(psnrafter0)
        self.validRecord["SSIM"].append(ssim0)
        self.validRecord["SSIMafter"].append(ssimafter0)
        
    def log(self,msg):
        f = open(self.logPath+"/generalLog.txt","a+")
        timestamp = datetime.datetime.now().strftime("[%m-%d  %H:%M:%S] ")
        f.write(timestamp + msg + "\n")
        f.close()
        
    def logNet(self,net):
        f = open(self.logPath+"/netInfo.txt","a+")
        timestamp = datetime.datetime.now().strftime("[%m-%d  %H:%M:%S] ")
        f.write(timestamp + str(net) + "\n")
        f.close()
        
    def write_to_file(self,param,isCheckpoint):
        sio.savemat(self.logPath+"/trainLog.mat",self.trainRecord)
        sio.savemat(self.logPath+"/validLog.mat",self.validRecord)
        f = open(self.logPath+"/generalLog.txt","a+")
        timestamp = datetime.datetime.now().strftime("[%m-%d  %H:%M:%S] ")
        saveNet(param, self.weightPath, self.trained_epoch, self.save_offset, isCheckpoint)
        f.write(timestamp + "file updated epoch:%05d\n" % (self.trained_epoch))
        f.close()
    
    def load_from_file(self, net, expectedEpoch, isCheckpoint=False, inTrain = True):
        if(expectedEpoch == 0):
            return
        tmpTrainRecord = sio.loadmat(self.logPath+"/trainLog.mat")
        tmpValidRecord = sio.loadmat(self.logPath+"/validLog.mat")
        lastRecord1 = np.max(tmpTrainRecord['epoch'])
        lastRecord2 = np.max(tmpValidRecord['epoch'])
        #assert lastRecord1 == lastRecord2, "trainLog epoch not equal to vaildLog"
        #assert lastRecord1 == expectedEpoch, "trainLog epoch not equal to expected epoch"
        print(">>>Log Data Loaded!")
        print(">>>[Log] last/target epoch:%d/%d"%(lastRecord1,expectedEpoch))
        #print(">>>[ValidLog] last epoch:%d"%lastRecord2)
        
        # self.trainRecord["epoch"] = tmpTrainRecord["epoch"][0].tolist()
        # self.trainRecord["trainLoss"] = tmpTrainRecord["trainLoss"][0].tolist()
        
        # self.validRecord["epoch"] = tmpValidRecord["epoch"][0].tolist()
        # self.validRecord["validLoss"] = tmpValidRecord["validLoss"][0].tolist()
        # self.validRecord["PSNR"] = tmpValidRecord["PSNR"][0].tolist()
        # self.validRecord["PSNRafter"] = tmpValidRecord["PSNRafter"][0].tolist()

        if(expectedEpoch in tmpTrainRecord["epoch"]):
            for key in self.trainRecord.keys():
                self.trainRecord[key] = tmpTrainRecord[key][0].tolist()
            index = self.trainRecord["epoch"].index(expectedEpoch)
            for key in self.trainRecord.keys():
                #self.trainRecord[key] = tmpTrainRecord[key][0].tolist()
                self.trainRecord[key] = self.trainRecord[key][:index+1]
        else:
            assert False, "No expectedEpoch in trainRecord"
        if(expectedEpoch in tmpValidRecord["epoch"]):
            for key in self.validRecord.keys():
                self.validRecord[key] = tmpValidRecord[key][0].tolist()
            index = self.validRecord["epoch"].index(expectedEpoch)
            for key in self.validRecord.keys():
                #self.validRecord[key] = tmpValidRecord[key][0].tolist()
                self.validRecord[key] = self.validRecord[key][:index+1]
        else:
            assert False, "No expectedEpoch in validRecord"
 
        
        loadNet(net, self.weightPath, expectedEpoch, isCheckpoint)
        self.trained_epoch = expectedEpoch
        if(inTrain):
            self.log(".mat file load from epoch:%05d\n" % (expectedEpoch))