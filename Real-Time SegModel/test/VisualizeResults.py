import numpy as np
import torch
from torch.autograd import Variable
import glob
import cv2
from PIL import Image as PILImage
from models import Espnet as Net
import os
import time
from argparse import ArgumentParser
import json
import pickle
import torch.backends.cudnn as cudnn
from test.IOUEval import iouEval
import scipy.misc as m
from data_loader import DataSet
cudnn.benchmark = False
from models.Model import get_model

pallete = [128, 64, 128,
           244, 35, 232,
           70, 70, 70,
           102, 102, 156,
           190, 153, 153,
           153, 153, 153,
           250, 170, 30,
           220, 220, 0,
           107, 142, 35,
           152, 251, 152,
           70, 130, 180,
           220, 20, 60,
           255, 0, 0,
           0, 0, 142,
           0, 0, 70,
           0, 60, 100,
           0, 80, 100,
           0, 0, 230,
           119, 11, 32,
           0, 0, 0]


def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    '''
    img[img == 19] = 255
    img[img == 18] = 33
    img[img == 17] = 32
    img[img == 16] = 31
    img[img == 15] = 28
    img[img == 14] = 27
    img[img == 13] = 26
    img[img == 12] = 25
    img[img == 11] = 24
    img[img == 10] = 23
    img[img == 9] = 22
    img[img == 8] = 21
    img[img == 7] = 20
    img[img == 6] = 19
    img[img == 5] = 17
    img[img == 4] = 13
    img[img == 3] = 12
    img[img == 2] = 11
    img[img == 1] = 8
    img[img == 0] = 7
    img[img == 255] = 0
    return img


def main():
    #define the parameters
    # load config file
    model_path = '/home/zhengxiawu/work/real_time_seg'
    model_num = 90
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # load config
    config_file = os.path.join(model_path, 'config/RF_LW_resnet_152_camVid.json')
    config = json.load(open(config_file))

    # set file name
    data_dir = os.path.join(model_path, config['DATA']['data_dir'])
    data_cache_file = os.path.join(data_dir, config['DATA']['cached_data_file'])
    result_save_dir = os.path.join(model_path, 'result', config['name'])
    weight_file = '/home/zhengxiawu/work/real_time_seg/para/RF_LW_resnet_152_camVid_batch_size_2_multi_scale/best.pth'
    GPU = False
    assert os.path.isfile(weight_file),"no weight file!!!"

    # data hyper parameters
    classes = config['DATA']['classes']
    scale_in = config['DATA']['scale_in']
    img_suffix = config['DATA']['img_suffix']
    data_name = config['DATA']['name']


    if not os.path.exists(result_save_dir):
        os.mkdir(result_save_dir)

    # read all the images in the folder
    #image_list = glob.glob(val_data_dir + os.sep + '*/*.' + img_suffix)


    data = pickle.load(open(data_cache_file, "rb"))
    image_list = data['valIm']
    up = torch.nn.Upsample(scale_factor=scale_in, mode='bilinear')
    if GPU:
        up.cuda()
    model = get_model(config['MODEL']['name'], classes,mode='train')
    model.load_state_dict(torch.load(weight_file))
    if GPU:
        model.cuda()
    model.eval()
    total_time = 0
    for i, imgName in enumerate(image_list):
        img = cv2.imread(imgName).astype(np.float32)
        for j in range(3):
            img[:, :, j] -= data['mean'][j]
        for j in range(3):
            img[:, :, j] /= data['std'][j]

        # resize the image to 1024x512x3
        img = cv2.resize(img, (1024, 512))
        #img = cv2.resize(img, (2048, 1024))
        img /= 255
        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension
        with torch.no_grad():
            img_variable = Variable(img_tensor)
        if GPU:
            img_variable = img_variable.cuda()
        torch.cuda.synchronize()
        time_start = time.time()
        img_out = model(img_variable)
        time_end = time.time()
        torch.cuda.synchronize()

        total_time += (time_end-time_start)
        print time_end-time_start

        if scale_in > 1:
            img_out = up(img_out)
        if GPU:
            classMap_numpy = img_out[0].max(0)[1].byte().cpu().data.numpy()
        else:
            classMap_numpy = img_out[0].max(0)[1].byte().data.numpy()

        if i % 100 == 0:
            print(i)

        name = imgName.split('/')[-1]
        if data_name == 'cityscape':
            classMap_numpy = relabel(classMap_numpy.astype(np.uint8))
            classMap_numpy = cv2.resize(classMap_numpy, (2048, 1024), interpolation=cv2.INTER_NEAREST)
        else:
            classMap_numpy_color = PILImage.fromarray(classMap_numpy)
            classMap_numpy_color.putpalette(pallete)
            classMap_numpy_color.save(result_save_dir + os.sep + 'c_' + name.replace(img_suffix, 'png'))

        #cv2.imwrite(result_save_dir + os.sep + name.replace(img_suffix, 'png'), classMap_numpy)

    print 'inference time is:'+str(float(total_time)/float(len(image_list)))
    print 'done'

if __name__ == '__main__':

    main()
