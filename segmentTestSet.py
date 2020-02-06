location_of_NetworkTraining_py="../"
import sys
sys.path.append(location_of_NetworkTraining_py)
import numpy as np
import torch
from net_v2 import UNet2d
import os
from skimage.io import imread, imsave
from skimage.transform import rescale
from NetworkTraining_py import f1

datadir="/"
exec(open("testFiles.txt").read())

log_dir="log_v1"
net = UNet2d().cuda()
saved_net=torch.load(os.path.join(log_dir,"net_Test_bestF1.pth")) #
net.load_state_dict(saved_net['state_dict'])
net.eval();

out_dir="output_best_v1" 

def process_output(o):
    e=np.exp(o[0,1,:,:])
    prob=e/(e+1)
    return prob

def prepareimg(img):
    maxv=6200.0
    minv=5700.0
    rang=maxv-minv
    img=(img.astype(np.float)-minv)/rang
    return img
  
outdir=os.path.join(log_dir,out_dir)
os.makedirs(outdir)

imgs=[]
lbls=[]
outs=[]
for f in testFiles: 
  bn=os.path.splitext(os.path.basename(f[0]))[0]
  img=imread(os.path.join(datadir,f[0]),plugin='tifffile')
  lbl=imread(os.path.join(datadir,f[1]))
  img = prepareimg(img).astype(np.float32)
  print(img.shape)
  inp=img.reshape(1,1,img.shape[-2],img.shape[-1])
  with torch.no_grad():
    oup=net.forward(torch.from_numpy(inp).cuda())
  prob=process_output(oup).cpu().numpy()
  outs.append(prob)
  lbls.append(lbl)
  imgs.append(img)
  #np.save(os.path.join(outdir,bn),prob.cpu().numpy())
  imsave(os.path.join(outdir,bn+".png"),prob)

p1,r1=f1.PRFromOutputsAndGroundTruths(imgs,lbls)
p2,r2=f1.PRFromOutputsAndGroundTruths(outs,lbls)
prfile=open(outdir+"_pr.txt","w")
for a,b,c,d in zip(p1,r1,p2,r2):
  prfile.write("{}\t{}\t{}\t{}\n".format(a,b,c,d))
prfile.close()
