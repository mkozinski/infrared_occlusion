location_of_NetworkTraining_py="../"
import sys
sys.path.append(location_of_NetworkTraining_py)
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from net_v2 import UNet2d
from NetworkTraining_py.loggerBasic import LoggerBasic
from NetworkTraining_py.loggerF1 import LoggerF1
from NetworkTraining_py.dataset import Dataset
from NetworkTraining_py.datasetCrops import TestDataset
from NetworkTraining_py.crop import crop
import os
import os.path
import torch.nn.functional as F
from shutil import copyfile
from NetworkTraining_py.trainer import trainer
from NetworkTraining_py.tester import tester
from random import randint, random
import math
from skimage.io import imread
from skimage.transform import rescale

def countClasses(lbls):
  count=np.array([0,0])
  for lbl in lbls:
    count[0]+=np.equal(lbl,0).sum()
    count[1]+=np.equal(lbl,1).sum()
  return count

def calcClassWeights(count):
  # reweight gently - do not balance completely
  freq=count.astype(np.double)/count.sum()
  freq+=1.02
  freq=np.log(freq)
  w=np.power(freq,-1)
  w=w/w.sum()
  return torch.Tensor(w)

def augmentTrain(img,lbl,cropSz,njitter):

  # random crop 
  maxstartind1=lbl.shape[0]-cropSz[0]
  maxstartind2=lbl.shape[1]-cropSz[1]
  if maxstartind1>0:
    startind1=randint(0,maxstartind1)
  else:
    startind1=maxstartind1-njitter//2+randint(0,njitter)
  if maxstartind2>0:
    startind2=randint(0,maxstartind2)
  else:
    startind2=maxstartind2-njitter//2+randint(0,njitter)
  cropinds=[slice(startind1,startind1+cropSz[0]),
            slice(startind2,startind2+cropSz[1]),]
  img,_=crop(img,cropinds)
  lbl,_=crop(lbl,cropinds,fill=255)

  # flip
  if random()>0.5 :
    img=np.flip(img,0)
    lbl=np.flip(lbl,0)
  if random()>0.5 :
    img=np.flip(img,1)
    lbl=np.flip(lbl,1)
  
  # axis swap
  if random()>0.5 :
    img=np.swapaxes(img,0,1)
    lbl=np.swapaxes(lbl,0,1)

  # copy the np array to avoid negative strides resulting from flip
  it=torch.from_numpy(np.copy(img).astype(np.float32)).unsqueeze(0)
  lt=torch.from_numpy(lbl.astype(np.long))

  return it,lt

def preproc(img,lbl):
  # this function is used in the training loop, and not in the data loader
  # this is needed to avoid transfer of data to GPU in threads of the dataloader
  return img.cuda(), lbl.cuda()

def augmentTest(img,lbl):
  lbl=torch.from_numpy(lbl)
  img=torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
  return img,lbl

def test_preproc(output,target):
  # preprocessing for test logger
  idx=torch.Tensor.mul_(target<=1, target>=0).reshape(target.numel())
  o=output[:,1,:,:]
  oo=o.reshape(target.numel())[idx]
  t=target.reshape(target.numel())[idx]
  o=torch.pow(torch.exp(-oo)+1,-1)
  return o,t

log_dir="log_v1"
datadir="/"
exec(open("trainFiles.txt").read())
exec(open("testFiles.txt").read())
def prepareimg(img):
    maxv=6200.0
    minv=5700.0
    rang=maxv-minv
    img=(img.astype(np.float)-minv)/rang
    return img
trainimgs=[]
trainlbls=[]
for f in trainFiles:
  img =imread(os.path.join(datadir,f[0]),plugin='tifffile')
  img = img.astype(np.float32)
  img=prepareimg(img)
  lbl =imread(os.path.join(datadir,f[1])).astype(np.uint8)
  trainimgs.append(img)
  trainlbls.append(lbl)
testimgs=[]
testlbls=[]
for f in testFiles:
  img =imread(os.path.join(datadir,f[0]),plugin='tifffile')
  img =img.astype(np.float32)
  img =prepareimg(img)
  lbl =imread(os.path.join(datadir,f[1])).astype(np.uint8)
  testimgs.append(img)
  testlbls.append(lbl)

os.makedirs(log_dir)
copyfile(__file__,os.path.join(log_dir,"setup.txt"))

train_dataset=Dataset(trainimgs,trainlbls,
                 lambda i,l: augmentTrain(i,l,np.array([480,480]),0))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=12,
                  shuffle=True, num_workers=12, drop_last=True)

test_dataset = TestDataset(testimgs,testlbls,np.array([512,512]),
                  [0,0], augmentTest, ignoreInd=255)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                  shuffle=False, num_workers=1, drop_last=False)

net = UNet2d().cuda()
#prev_log_dir="log_v2"
#saved_net=torch.load(os.path.join(prev_log_dir,"net_last.pth"))
#net.load_state_dict(saved_net['state_dict'])
net.train()

weight=calcClassWeights(countClasses(train_dataset.lbl))
loss = torch.nn.CrossEntropyLoss(weight=weight,ignore_index=255).cuda()
print("loss.weight",loss.weight)

optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-4)

logger= LoggerBasic(log_dir,"Basic",100)

logger_test=LoggerF1(log_dir,"Test",test_preproc, nBins=10000, saveBest=True)
tstr=tester(test_loader,logger_test,preproc)

lr_lambda=lambda e: 1/(1+e*1e-5)
lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
trn=trainer(net, train_loader, optimizer, loss, logger, tstr, 100,
            lr_scheduler=lr_scheduler,preprocImgLbl=preproc)

if __name__ == '__main__':
  print(log_dir)
  trn.train(100000)
