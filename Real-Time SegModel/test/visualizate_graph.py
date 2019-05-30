from torch.autograd import Variable
import torch.onnx
import torchvision
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
model_name = 'ESpnet_2_8'
dummpy_input = Variable(torch.randn((10,3,256,256)).cuda())
if model_name == 'ESpnet_2_8':
    from models import Espnet
    model = Espnet.ESPNet_Encoder(20, 2, 8)
    model.cuda()

torch.onnx.export(model,dummpy_input,"/home/zhengxiawu/work/real_time_seg/para/Espnet_cityscape/espnet.onnx")