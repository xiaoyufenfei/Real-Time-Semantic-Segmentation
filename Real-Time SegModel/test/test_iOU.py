import os
import json
import torch
import time
import pickle
import data_loader.DataSet as  myDataLoader
import data_loader.Transforms as myTransforms
from data_loader import loadData
from test.myIOUEval import iouEval
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utils import VisualizeGraph as viz
from models.Criteria import CrossEntropyLoss2d
from models.Model import get_model
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def val(classes, val_loader, model, criterion,up = None, ignore_label = []):
    '''
    :param args: general arguments
    :param val_loader: loaded for validation dataset
    :param model: model
    :param criterion: loss function
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    #switch to evaluation mode
    model.eval()

    iouEvalVal = iouEval(classes,len(val_loader),ignore_label)

    total_time = 0
    total_batches = len(val_loader)
    for i, (input, target) in enumerate(val_loader):

        with torch.no_grad():
            img_variable = Variable(input)
            target = target.cuda()
        img_variable = img_variable.cuda()
        target_var = target.cuda()
        # input = input.cuda()
        # target = target.cuda()
        # with torch.no_grad():
        #     input_var = torch.autograd.Variable(input)
        #     target_var = torch.autograd.Variable(target)

        # run the mdoel
        torch.cuda.synchronize()
        time_start = time.time()
        output = model(img_variable)
        time_end = time.time()
        torch.cuda.synchronize()
        time_taken = time_end - time_start
        total_time += time_taken
        if up is not None:
            output = up(output)
        # compute the confusion matrix
        iouEvalVal.addBatch(output.max(1)[1].data, target_var.data)

        print('[%d/%d] time: %.16f' % (i, total_batches,  time_taken))

    print ('total time is:'+str(float(total_time)/float(total_batches)))


    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalVal.getMetric()

    return  overall_acc, per_class_acc, per_class_iu, mIOU


if __name__ == '__main__':
    #load config file
    model_path = '/home/zhengxiawu/work/real_time_seg'
    #load config
    config_file = os.path.join(model_path, 'config/ESPnet_decoder_cityscape.json')
    weight_file = '/home/zhengxiawu/work/real_time_seg/pretrained/decoder/espnet_p_2_q_8.pth'
    mode = 'test'
    config = json.load(open(config_file))

    #set file name
    data_dir = os.path.join(model_path,config['DATA']['data_dir'])
    data_cache_file = os.path.join(data_dir,config['DATA']['cached_data_file'])
    save_dir = os.path.join(model_path, 'para', config['name'])+'/'

    #data hyper parameters
    classes = config['DATA']['classes']
    width = config['DATA']['width']
    height = config['DATA']['height']
    scales = config['DATA']['train_args']['scale']
    random_crop_size = config['DATA']['train_args']['random_crop_size']
    scale_in = config['DATA']['scale_in']
    val_scale = config['DATA']['val_args']['scale']
    batch_size = config['DATA']['train_args']['batch_size']
    data_name = config['DATA']['name']
    ignore_label = config['DATA']['ignore_label']

    #network hyper parameters
    lr = config['lr']
    lr_step = config['lr_step']
    save_step = config['save_step']
    if scale_in > 1:
        up = torch.nn.Upsample(scale_factor=scale_in, mode='bilinear')
        up.cuda()
    else:
        up = None
    #set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    #load the dataset

    data = pickle.load(open(data_cache_file, "rb"))

    data['name'] = data_name
    #get model
    model = get_model(config['MODEL']['name'], classes,mode='test')
    model.load_state_dict(torch.load(weight_file))
    model.cuda()
    model.eval()

    # define optimization criteria
    weight = torch.from_numpy(data['classWeights'])  # convert the numpy array to torch
    weight = weight.cuda()
    criteria = CrossEntropyLoss2d()  # weight


    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(width, height),
        myTransforms.ToTensor(1),
        #
    ])
    val_data_loader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['valIm'], data['valAnnot'], transform=valDataset,data_name=data['name']),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    cudnn.benchmark = True
    start_epoch = 0

    overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val = val(classes, val_data_loader, model,criteria,up,ignore_label)
    print mIOU_val
    print per_class_iu_val