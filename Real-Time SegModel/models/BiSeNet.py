import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class conv_block(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=3,stride=2,padding=1):
        super(conv_block,self).__init__()
        self.conv=nn.Conv2d(in_channel,out_channel,kernel,stride,padding)
        self.bn=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU()

    def forward(self,x):
        x=self.relu(self.bn(self.conv(x)))
        return x

class spatial_path(nn.Module):
    def __init__(self,in_channel1,in_channel2,in_channel3,out_channel):
        super(spatial_path,self).__init__()
        self.out_channel=out_channel
        self.conv_block1=conv_block(in_channel1,in_channel2)
        self.conv_block2=conv_block(in_channel2,in_channel3)
        self.conv_block3=conv_block(in_channel3,out_channel)

    def forward(self,x):
        x=self.conv_block1(x)
        x=self.conv_block2(x)
        x=self.conv_block3(x)
        return x

class ARM(nn.Module):
    def __init__(self,in_channel):
        super(ARM,self).__init__()
        self.conv=nn.Conv2d(in_channel,in_channel,1)
        self.bn=nn.BatchNorm2d(in_channel)
        self.relu=nn.ReLU()

    def forward(self,x):
        out=torch.mean(x,2,keepdim=True)
        out=torch.mean(out,3,keepdim=True)
        out=self.conv(out)
        out=self.bn(out)
        out=torch.sigmoid(out)
        out=torch.mul(x,out)

class context_path(nn.Module):
    def __init__(self,mode='resnet18'):
        super(context_path,self).__init__()
        if mode=='resnet18':
            model=torchvision.models.resnet18(pretrained=True)
        elif mode=='resnet101':
            model = torchvision.models.resnet101(pretrained=True)
        self.conv1=model.conv1
        self.bn=model.bn1
        self.relu=model.relu
        self.maxpool=model.maxpool
        self.initial=nn.Sequential(self.conv1,self.bn,self.relu,self.maxpool)
        self.layer1=model.layer1
        self.layer2=model.layer2
        self.layer3=model.layer3
        self.layer4=model.layer4
        self.out_channel=model.layer4[1].conv2.out_channels+model.layer3[1].conv2.out_channels

    def forward(self,x):
        x=self.initial(x)
        x=self.layer1(x)
        x=self.layer2(x)
        out1=self.layer3(x)
        out2=self.layer4(out1)
        tail=torch.mean(out2,2,keepdim=True)
        tail=torch.mean(tail,3,keepdim=True)
        return out1,out2,tail

class FFM(nn.Module):
    def __init__(self,in_channel,out_class):
        super(FFM,self).__init__()
        self.conv1=conv_block(in_channel,out_class,stride=1)
        self.conv2=nn.Conv2d(out_class,out_class,1)
        self.conv3=nn.Conv2d(out_class,out_class,1)
        self.relu=nn.ReLU()

    def forward(self,x1,x2):
        out=torch.cat((x1,x2),1)
        out=self.conv1(out)
        w=torch.mean(out,2,keepdim=True)
        w = torch.mean(w, 3, keepdim=True)
        w=self.conv2(w)
        w=self.relu(w)
        w=self.conv2(w)
        w=torch.sigmoid(w)
        #out=F.mul(out,w)+out
        out = out * w + out
        return out

class BiSeNet(nn.Module):
    def __init__(self,mode='resnet18',out_class=3):
        super(BiSeNet,self).__init__()
        self.conv=nn.Conv2d(out_class,out_class,1)
        self.spatial_path=spatial_path(3,64,128,256)
        self.context_path=context_path(mode='resnet18')
        self.FFM=FFM(self.spatial_path.out_channel+self.context_path.out_channel,out_class)

    def forward(self,x):
        sx=self.spatial_path(x)
        cx1,cx2,tail=self.context_path(x)
        cx1=nn.functional.interpolate(cx1, scale_factor=2, mode='bilinear')
        cx2=F.interpolate(cx2, scale_factor=4, mode='bilinear')
        #print(cx1.shape)
        #print(cx2.shape)
        cx=torch.cat((cx1,torch.mul(cx2,tail)),1)
        print(cx.shape)
        print(sx.shape)
        out=self.FFM(cx,sx)
        out=F.interpolate(out,scale_factor=8,mode='bilinear')
        out=self.conv(out)
        return out

if __name__=='__main__':
    #x=torch.rand(1,3,512,1024)
    x1 = torch.rand(1, 3, 544, 736)
    x2 = torch.rand(1, 3, 456, 600)
    x3 = torch.rand(1, 3, 272, 360)
    x4 = torch.rand(1, 3, 360, 480)
    model=BiSeNet()
    model.eval()
    for i in [x1,x2,x3,x4]:
        y=model(i)
        print(y.shape)




