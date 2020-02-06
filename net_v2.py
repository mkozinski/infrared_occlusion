
# receptive field 380x380
# additional global pooling in the middle

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn_relu(in_channels, out_channels, kernel_size=3):
    stride=1
    padding=kernel_size//2
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
    )

class upResBlock(nn.Module):
  def __init__(self, in_channels, kernel_size=3):
    super().__init__()
    self.cbr=conv_bn_relu(in_channels,2*in_channels,kernel_size)
  def forward(self, x):
    return self.cbr(x)

class dwResBlock(nn.Module):
  def __init__(self, in_channels, kernel_size=3):
    super().__init__()
    self.cbr=conv_bn_relu(in_channels,in_channels//2,kernel_size)
  def forward(self, x):
    return self.cbr(x)

def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class globalization(nn.Module):
    def __init__(self,nChannels):
        super(globalization, self).__init__()
        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.net=nn.Sequential(
            nn.Conv2d(2*nChannels,2*nChannels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*nChannels,  nChannels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.net(torch.cat([x,self.pool(x).expand_as(x)],dim=1))

class UNet2d(nn.Module):
    def __init__(self):
        super().__init__()
        input_channels = 1
        nclasses = 1
        # go down
        self.conv0 = conv_bn_relu(input_channels, 64)
        self.conv1 = upResBlock(64) 
        self.conv2 = upResBlock(128) 
        self.conv3 = upResBlock(256) 
        self.conv4 = upResBlock(512) 
        self.conv5 = upResBlock(1024) 

        self.g=globalization(2048)

        self.down_pooling = nn.MaxPool2d(2)

        # go up
        self.up_pool5 = up_pooling(2048, 1024)
        self.deconv5 = dwResBlock(2048) 
        self.up_pool4 = up_pooling(1024, 512)
        self.deconv4 = dwResBlock(1024) 
        self.up_pool3 = up_pooling(512, 256)
        self.deconv3 = dwResBlock(512) 
        self.up_pool2 = up_pooling(256, 128)
        self.deconv2 = dwResBlock(256) 
        self.up_pool1 = up_pooling(128, 64)
        self.deconv1 = dwResBlock(128) 

        self.conv_final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # normalize input data
        # go down
        x0 = self.conv0(x)
        p0 = self.down_pooling(x0)
        x1 = self.conv1(p0)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)

        xg = self.g(x5)

        # go up
        u5 = self.up_pool5(xg)
        c5 = torch.cat([u5, x4], dim=1)
        y5 = self.deconv5(c5)

        u4 = self.up_pool4(y5)
        c4 = torch.cat([u4, x3], dim=1)
        y4 = self.deconv4(c4)

        u3 = self.up_pool3(y4)
        c3 = torch.cat([u3, x2], dim=1)
        y3 = self.deconv3(c3)

        u2 = self.up_pool2(y3)
        c2 = torch.cat([u2, x1], dim=1)
        y2 = self.deconv2(c2)

        u1 = self.up_pool1(y2)
        c1 = torch.cat([u1, x0], dim=1)
        y1 = self.deconv1(c1)


        output = F.pad(self.conv_final(y1), [0,0, 0,0, 1,0])
        return output
