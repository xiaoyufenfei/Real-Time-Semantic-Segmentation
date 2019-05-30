import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class BR(nn.Module):
    def __init__(self, nOut):
        super(BR,self).__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)
    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output
class CBR(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1,dilated = 1):
        super(CBR,self).__init__()
        padding = int((kSize - 1)/2)*dilated
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=dilated)
        self.BR = BR(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.BR(output)
        return output
class upBR(nn.Module):
    def __init__(self, nIn, nOut):
        super(upBR,self).__init__()
        self.conv = nn.ConvTranspose2d(nIn, nOut, 2, stride=2, padding=0, output_padding=0, bias=False)
        self.BR = BR(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.BR(output)
        return output


class Basenet(nn.Module):
    def __init__(self, num_classes=20):
        super(Basenet, self).__init__()
        #downsample block 1
        self.downsample_1 = CBR(3,16,3,2)
        self.downsample_2 = CBR(16,64,3,2)
        self.regular_1 = nn.ModuleList()
        for i in range(5):
            self.regular_1.append(CBR(64,64,3,1))
        self.downsample_3 = CBR(64,128,3,2)
        self.regular_2 = nn.ModuleList()
        for i in range(8):
            self.regular_2.append(CBR(128, 128, 3, 1))
        self.Upsample_1 = upBR(128,64)
        self.regular_3 = nn.ModuleList()
        for i in range(2):
            self.regular_3.append(CBR(64, 64, 3, 1))
        self.Upsample_2 = upBR(64, 2*num_classes)
        self.regular_4 = nn.ModuleList()
        for i in range(2):
            self.regular_4.append(CBR(2*num_classes, 2*num_classes, 3, 1))
        self.Upsample_3 = upBR(2 * num_classes, num_classes)
        self.weights_init()

    def weights_init(self):
        for idx, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        downsample_1 = self.downsample_1(x)#/2
        downsample_2 = self.downsample_2(downsample_1)#/4
        x = downsample_2
        for regular_layer in self.regular_1:
            x = regular_layer(x)
        regular_1 = x
        downsample_3 = self.downsample_3(regular_1)#/8
        x = downsample_3
        for regular_layer in self.regular_2:
            x = regular_layer(x)
        regular_2 = x
        up_sample_1 = self.Upsample_1(regular_2)#/4
        x = up_sample_1
        for regular_layer in self.regular_3:
            x = regular_layer(x)
        regular_3 = x
        up_sample_2 = self.Upsample_2(regular_3)#/2
        x = up_sample_2
        for regular_layer in self.regular_4:
            x = regular_layer(x)
        regular_4 = x
        up_sample_3 = self.Upsample_3(regular_4)#/1
        return up_sample_3

if __name__ == '__main__':

    x1 = torch.rand(1, 3, 544, 736)
    x2 = torch.rand(1, 3, 456, 600)
    x3 = torch.rand(1, 3, 272, 360)
    x4 = torch.rand(1, 3, 360, 480)
    model = Basenet()
    model.eval()
    for i in [x1, x2, x3, x4]:
        y = model(i)
        print(y.shape)




