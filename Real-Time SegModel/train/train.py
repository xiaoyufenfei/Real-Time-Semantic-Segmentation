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



def val(classes, val_loader, model, criterion,ignore_label):
    '''
    :param args: general arguments
    :param val_loader: loaded for validation dataset
    :param model: model
    :param criterion: loss function
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    #switch to evaluation mode
    model.eval()
    epoch_loss = []
    total_batches = len(val_loader)
    iouEvalVal = iouEval(classes,total_batches,ignore_label)
    for i, (input, target) in enumerate(val_loader):
        start_time = time.time()

        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # run the mdoel
        output = model(input_var)

        # compute the loss
        loss = criterion(output, target_var)

        epoch_loss.append(loss.item())

        time_taken = time.time() - start_time
        # compute the confusion matrix
        iouEvalVal.addBatch(output.max(1)[1].data, target_var.data)

        print('[%d/%d] loss: %.3f time: %.2f' % (i, total_batches, loss.item(), time_taken))

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)

    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalVal.getMetric()

    return average_epoch_loss_val, overall_acc, per_class_acc, per_class_iu, mIOU

def train(classes, train_loader, model, criterion, optimizer, epoch):
    '''
    :param train_loader: loaded for training dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    # switch to train mode
    model.train()



    epoch_loss = []

    total_batches = len(train_loader)
    iouEvalTrain = iouEval(classes,total_batches)
    for i, (input, target) in enumerate(train_loader):
        start_time = time.time()

        input = input.cuda()
        target = target.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        #run the mdoel
        output = model(input_var)

        #set the grad to zero
        optimizer.zero_grad()
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        time_taken = time.time() - start_time

        #compute the confusion matrix
        iouEvalTrain.addBatch(output.max(1)[1].data, target_var.data)

        print('[%d/%d] loss: %.3f time:%.2f' % (i, total_batches, loss.item(), time_taken))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalTrain.getMetric()

    return average_epoch_loss_train, overall_acc, per_class_acc, per_class_iu, mIOU

def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    '''
    helper function to save the checkpoint
    :param state: model state
    :param filenameCheckpoint: where to save the checkpoint
    :return: nothing
    '''
    torch.save(state, filenameCheckpoint)

def netParams(model):
    '''
    helper function to see total network parameters
    :param model: model
    :return: total network parameters
    '''
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters

def multi_scale_loader(scales,random_crop_size,scale_in, batch_size, data):
    #input check
    assert len(scales) == len(random_crop_size), "the length of scales and random_crop_size should be same!!!"

    #transform
    data_loader = []
    for i,scale in enumerate(scales):
        if random_crop_size[i]>0:
            this_transform = myTransforms.Compose([
                myTransforms.Normalize(mean=data['mean'], std=data['std']),
                myTransforms.Scale(scale[0], scale[1]),
                myTransforms.RandomCropResize(random_crop_size[i]),
                myTransforms.RandomFlip(),
                #myTransforms.RandomCrop(64).
                myTransforms.ToTensor(scale_in),
                #
                ])
        else:
            this_transform = myTransforms.Compose([
                myTransforms.Normalize(mean=data['mean'], std=data['std']),
                myTransforms.Scale(scale[0], scale[1]),
                myTransforms.RandomFlip(),
                # myTransforms.RandomCrop(64).
                myTransforms.ToTensor(scale_in),
                #
            ])
        data_loader.append(
            torch.utils.data.DataLoader(myDataLoader.MyDataset(
                   data['trainIm'],
                   data['trainAnnot'],
                   transform=this_transform,data_name=data['name']),
                   batch_size=batch_size,
                   shuffle=True,
                   num_workers=8,
                   pin_memory=True))
    return data_loader
if __name__ == '__main__':
    #load config file
    model_path = '/home/zhengxiawu/work/real_time_seg'
    #load config
    config_file = os.path.join(model_path, 'config/Basenet_camVid.json')
    config = json.load(open(config_file))

    #set file name
    data_dir = os.path.join(model_path,config['DATA']['data_dir'])
    data_cache_file = os.path.join(data_dir,config['DATA']['cached_data_file'])
    save_dir = os.path.join(model_path, 'para', config['name'])+'/'
    assert not os.path.isfile(os.path.join(save_dir,'best.pth'))

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

    #set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_num'])
    #load the dataset
    if not os.path.isfile(data_cache_file):
        dataLoad = loadData.LoadData(data_dir,config['DATA']['classes'],data_cache_file,dataset=data_name)
        data = dataLoad.processData()
        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(data_cache_file, "rb"))

    data['name'] = data_name

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(width, height),
        myTransforms.ToTensor(scale_in),
        #
    ])
    val_data_loader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['valIm'], data['valAnnot'], transform=valDataset,data_name=data_name),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    for i, (input, target) in enumerate(val_data_loader):
        start_time = time.time()

        input = input.cuda()
        target = target.cuda()
    #get model
    model = get_model(config['MODEL']['name'],classes)

    model = model.cuda()

    # create the directory if not exist
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    #visulize
    if config['visulize']:
        x = Variable(torch.randn(1, 3, width, height))

        x = x.cuda()

        y = model.forward(x)
        g = viz.make_dot(y)
        g.render(save_dir + 'model.png', view=False)

    #caculate the parameters
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))

    # define optimization criteria
    weight = torch.from_numpy(data['classWeights'])  # convert the numpy array to torch
    weight = weight.cuda()
    criteria = CrossEntropyLoss2d(weight)  # weight

    print('Data statistics')
    print(data['mean'], data['std'])
    print(data['classWeights'])

    #get data_loaders
    train_data_loaders = multi_scale_loader(scales,random_crop_size,scale_in, batch_size, data)


    cudnn.benchmark = True
    start_epoch = 0

    #load resume
    if config['resume']:
        if os.path.isfile(config['resumeLoc']):
            print("=> loading checkpoint '{}'".format(config['resume']))
            checkpoint = torch.load(config['resumeLoc'])
            start_epoch = checkpoint['epoch']
            #args.lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(config['resume'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config['resume']))

    #log file
    logFileLoc = os.path.join(save_dir,'log.txt')
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t" % ('Epoch', 'Loss(Tr)', 'Loss(val)', 'mIOU (tr)', 'mIOU (val'))
    logger.flush()

    optimizer = torch.optim.Adam(model.parameters(), lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)
    # we step the loss by 2 after step size is reached
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.5)

    best_mIOU = -100
    for epoch in range(start_epoch, config['num_epoch']):

        scheduler.step(epoch)
        lr = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " + str(lr))

        # train for one epoch
        # We consider 1 epoch with all the training data (at different scales)
        for i in train_data_loaders:
            lossTr, overall_acc_tr, per_class_acc_tr, per_class_iu_tr, mIOU_tr = train(classes,i,model, criteria, optimizer, epoch)

        # evaluate on validation set
        lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val = val(classes, val_data_loader, model, criteria,ignore_label)

        #save check point
        # if (epoch+1)%save_step == 0:
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': str(model),
        #         'state_dict': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lossTr': lossTr,
        #         'lossVal': lossVal,
        #         'iouTr': mIOU_tr,
        #         'iouVal': mIOU_val,
        #         'lr': lr
        #     }, save_dir + 'checkpoint.pth.tar')
        #     # save the model also
        #     model_file_name = save_dir + 'model_' + str(epoch + 1) + '.pth'
        #     torch.save(model.state_dict(), model_file_name)
        #     with open(save_dir + 'acc_' + str(epoch) + '.txt', 'w') as log:
        #         log.write(
        #             "\nEpoch: %d\t Overall Acc (Tr): %.4f\t Overall Acc (Val): %.4f\t mIOU (Tr): %.4f\t mIOU (Val): %.4f" % (
        #             epoch, overall_acc_tr, overall_acc_val, mIOU_tr, mIOU_val))
        #         log.write('\n')
        #         log.write('Per Class Training Acc: ' + str(per_class_acc_tr))
        #         log.write('\n')
        #         log.write('Per Class Validation Acc: ' + str(per_class_acc_val))
        #         log.write('\n')
        #         log.write('Per Class Training mIOU: ' + str(per_class_iu_tr))
        #         log.write('\n')
        #         log.write('Per Class Validation mIOU: ' + str(per_class_iu_val))

        #save the best
        if best_mIOU < mIOU_val:
            best_mIOU = mIOU_val
            model_file_name = save_dir + 'best.pth'
            torch.save(model.state_dict(), model_file_name)
            with open(save_dir + 'best.txt', 'w') as log:
                log.write(
                    "\nEpoch: %d\t Overall Acc (Tr): %.4f\t Overall Acc (Val): %.4f\t mIOU (Tr): %.4f\t mIOU (Val): %.4f" % (
                        epoch, overall_acc_tr, overall_acc_val, mIOU_tr, mIOU_val))
                log.write('\n')
                log.write('Per Class Training Acc: ' + str(per_class_acc_tr))
                log.write('\n')
                log.write('Per Class Validation Acc: ' + str(per_class_acc_val))
                log.write('\n')
                log.write('Per Class Training mIOU: ' + str(per_class_iu_tr))
                log.write('\n')
                log.write('Per Class Validation mIOU: ' + str(per_class_iu_val))

        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f" % (epoch, lossTr, lossVal, mIOU_tr, mIOU_val, lr))
        logger.flush()
        print("Epoch : " + str(epoch) + ' Details')
        print("\nEpoch No.: %d\tTrain Loss = %.4f\tVal Loss = %.4f\t mIOU(tr) = %.4f\t mIOU(val) = %.4f" % (
        epoch, lossTr, lossVal, mIOU_tr, mIOU_val))
    logger.close()
    pass