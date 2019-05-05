import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import OrderedDict
from models.enet import ENet
from models.erfnet import ERFNet
from train import Train
from test import Test
from metric.iou import IoU
from args import get_arguments
from data.utils import enet_weighing, median_freq_balancing
import utils
from PIL import Image
import time
import numpy as np
from torchviz import make_dot, make_dot_from_trace
import matplotlib.pyplot as plt
import glob
import cv2

# Get the arguments
args = get_arguments()
use_cuda = args.cuda and torch.cuda.is_available()

def load_dataset(dataset):
    print("Loading dataset...")
    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

    image_transform = transforms.Compose( [transforms.Resize((args.height, args.width)), transforms.ToTensor()])

    label_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((args.height, args.width)), utils.PILToLongTensor()])

    train_set = dataset( args.dataset_dir, transform=image_transform, label_transform=label_transform)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    val_set = dataset(args.dataset_dir, mode='val', transform=image_transform, label_transform=label_transform)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = val_loader

    # Get encoding between pixel valus in label images and RGB colors
    class_encoding = train_set.color_encoding
    # Get number of classes to predict
    num_classes = len(class_encoding)
    # Print information for debugging
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))
    # Get a batch of samples to display
    if args.mode.lower() == 'test':
        images, labels = iter(test_loader).next()
    else:
        images, labels = iter(train_loader).next()
    print("Image size:", images.size())
    print("Label size:", labels.size())
    
    # Get class weights from the selected weighing technique
    print("Weighing technique:", args.weighing)
    # print("Computing class weights...") 
    # if args.weighing.lower() == 'enet':
    #     class_weights = enet_weighing(train_loader, num_classes)
    # elif args.weighing.lower() == 'mfb':
    #     class_weights = median_freq_balancing(train_loader, num_classes)
    # else:
    #     class_weights = None
    class_weights = np.array([ 0.0000,  3.9490, 13.2085,  4.2485, 36.9267, 34.0329, 30.3585, 44.1654,
        38.5243,  5.7159, 32.2182, 16.3313, 30.7760, 46.8776, 11.1293, 44.1730,
        44.8546, 44.9209, 47.9799, 41.5301])
    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float()
        # Set the weight of the unlabeled class to 0
        if args.ignore_unlabeled:
            ignore_index = list(class_encoding).index('unlabeled')
            class_weights[ignore_index] = 0
    
    print("Class weights:", class_weights)
    return (train_loader, val_loader, test_loader), class_weights, class_encoding 
    
def train(train_loader, val_loader, class_weights, class_encoding):
    print("Training...")
    num_classes = len(class_encoding)
    model = ERFNet(num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # Learning rate decay scheduler
    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs, args.lr_decay)

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None

    metric = IoU(num_classes, ignore_index=ignore_index)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        model, optimizer, start_epoch, best_miou, val_miou, train_miou, val_loss, train_loss = utils.load_checkpoint( model, optimizer, args.save_dir, args.name, True)
        print("Resuming from model: Start epoch = {0} | Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0
        val_miou = []
        train_miou = []
        val_loss = []
        train_loss = []
    
    # Start Training
    train = Train(model, train_loader, optimizer, criterion, metric, use_cuda)
    val = Test(model, val_loader, criterion, metric, use_cuda)
    
    for epoch in range(start_epoch, args.epochs):
        print(">> [Epoch: {0:d}] Training".format(epoch))
        lr_updater.step()
        epoch_loss, (iou, miou) = train.run_epoch(args.print_step)
        print(">> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, epoch_loss, miou))
        train_loss.append(epoch_loss)
        train_miou.append(miou)

        #preform a validation test
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))
            loss, (iou, miou) = val.run_epoch(args.print_step)
            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, loss, miou))
            val_loss.append(loss)
            val_miou.append(miou)
            # Print per class IoU on last epoch or if best iou
            if epoch + 1 == args.epochs or miou > best_miou:
                for key, class_iou in zip(class_encoding.keys(), iou):
                    print("{0}: {1:.4f}".format(key, class_iou))
            # Save the model if it's the best thus far
            if miou > best_miou:
                print("Best model thus far. Saving...")
                best_miou = miou
                utils.save_checkpoint(model, optimizer, epoch + 1, best_miou, val_miou, train_miou, val_loss, train_loss, args)

    return model, train_loss, train_miou, val_loss, val_miou

def test(model, test_loader, class_weights, class_encoding):
    print("Testing...")
    num_classes = len(class_encoding)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if use_cuda:
        criterion = criterion.cuda()

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # Test the trained model on the test set
    test = Test(model, test_loader, criterion, metric, use_cuda)

    print(">>>> Running test dataset")
    loss, (iou, miou) = test.run_epoch(args.print_step)
    class_iou = dict(zip(class_encoding.keys(), iou))

    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))
    # Print per class IoU
    for key, class_iou in zip(class_encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou))

def video():
    print('testing from video')
    cameraWidth = 1920
    cameraHeight = 1080
    cameraMatrix = np.matrix([[1.3878727764994030e+03, 0,    cameraWidth/2],
    [0,    1.7987055172413220e+03,   cameraHeight/2],
    [0,    0,    1]])
    
    distCoeffs = np.matrix([-5.8881725390917083e-01, 5.8472404395779809e-01,
    -2.8299599929891900e-01, 0])
    
    vidcap = cv2.VideoCapture('test_content/massachusetts.mp4')
    success = True
    i=0
    while success:
        success,img = vidcap.read()
        if i%1000 ==0:
            print("frame: ",i)
            if args.rmdistort:
                P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(cameraMatrix,distCoeffs,(cameraWidth,cameraHeight),None)
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(cameraMatrix, distCoeffs, np.eye(3), P, (1920,1080), cv2.CV_16SC2)
                img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            # img = img.convert('RGB')
            # cv2.imshow('',img)
            # cv2.waitKey(0)
            # img2 = Image.open(filename).convert('RGB')
            class_encoding = color_encoding = OrderedDict([
                    ('unlabeled', (0, 0, 0)),
                    ('road', (128, 64, 128)),
                    ('sidewalk', (244, 35, 232)),
                    ('building', (70, 70, 70)),
                    ('wall', (102, 102, 156)),
                    ('fence', (190, 153, 153)),
                    ('pole', (153, 153, 153)),
                    ('traffic_light', (250, 170, 30)),
                    ('traffic_sign', (220, 220, 0)),
                    ('vegetation', (107, 142, 35)),
                    ('terrain', (152, 251, 152)),
                    ('sky', (70, 130, 180)),
                    ('person', (220, 20, 60)),
                    ('rider', (255, 0, 0)),
                    ('car', (0, 0, 142)),
                    ('truck', (0, 0, 70)),
                    ('bus', (0, 60, 100)),
                    ('train', (0, 80, 100)),
                    ('motorcycle', (0, 0, 230)),
                    ('bicycle', (119, 11, 32)) ])

            num_classes = len(class_encoding)
            model_path = os.path.join(args.save_dir, args.name)
            checkpoint = torch.load(model_path)
            model = ERFNet(num_classes)
            model = model.cuda()
            model.load_state_dict(checkpoint['state_dict'])
            img = img.resize((args.width, args.height), Image.BILINEAR)
            start = time.time()
            images = transforms.ToTensor()(img)
            torch.reshape(images, (1, 3, args.width, args.height))
            images= images.unsqueeze(0)
            with torch.no_grad():
                images = images.cuda()
                predictions = model(images) 
                end = time.time()
                print('model speed:',int(1/(end - start)),"FPS")
                _, predictions = torch.max(predictions.data, 1)
                label_to_rgb = transforms.Compose([utils.LongTensorToRGBPIL(class_encoding),transforms.ToTensor()])
                color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
                end = time.time()
                print('model+transform:',int(1/(end - start)),"FPS")
                utils.imshow_batch(images.data.cpu(), color_predictions)
        i+=1

def single():
    print('Mode: Single')
    img = Image.open('test_content/example_01.png').convert('RGB')

    class_encoding = color_encoding = OrderedDict([
            ('unlabeled', (0, 0, 0)),
            ('road', (128, 64, 128)),
            ('sidewalk', (244, 35, 232)),
            ('building', (70, 70, 70)),
            ('wall', (102, 102, 156)),
            ('fence', (190, 153, 153)),
            ('pole', (153, 153, 153)),
            ('traffic_light', (250, 170, 30)),
            ('traffic_sign', (220, 220, 0)),
            ('vegetation', (107, 142, 35)),
            ('terrain', (152, 251, 152)),
            ('sky', (70, 130, 180)),
            ('person', (220, 20, 60)),
            ('rider', (255, 0, 0)),
            ('car', (0, 0, 142)),
            ('truck', (0, 0, 70)),
            ('bus', (0, 60, 100)),
            ('train', (0, 80, 100)),
            ('motorcycle', (0, 0, 230)),
            ('bicycle', (119, 11, 32)) ])


    num_classes = len(class_encoding)
    model = ERFNet(num_classes)
    model_path = os.path.join(args.save_dir, args.name)
    print('Loading model at:',model_path)
    checkpoint = torch.load(model_path)
    # model = ENet(num_classes)
    model = model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    img = img.resize((args.width, args.height), Image.BILINEAR)
    start = time.time()
    images = transforms.ToTensor()(img)
    torch.reshape(images, (1, 3, args.width, args.height))
    images= images.unsqueeze(0)
    with torch.no_grad():
        images = images.cuda()
        predictions = model(images) 
        end = time.time()
        print('model speed:',int(1/(end - start)),"FPS")
        _, predictions = torch.max(predictions.data, 1)
        label_to_rgb = transforms.Compose([utils.LongTensorToRGBPIL(class_encoding),transforms.ToTensor()])
        color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
        end = time.time()
        print('model+transform:',int(1/(end - start)),"FPS")
        utils.imshow_batch(images.data.cpu(), color_predictions)


if __name__ == '__main__':
    if args.mode.lower() == 'video':
        video()
    elif args.mode.lower() == 'single':
        single()
    else:
        # Fail fast if the saving directory doesn't exist
        assert os.path.isdir(            args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(args.dataset_dir)
        assert os.path.isdir(            args.save_dir), "The directory \"{0}\" doesn't exist.".format(args.save_dir)
        # Import the requested dataset
        if args.dataset.lower() == 'camvid':
            from data import CamVid as dataset
        elif args.dataset.lower() == 'cityscapes':
            from data import Cityscapes as dataset
        else:
            raise RuntimeError("\"{0}\" is not a supported dataset.".format(args.dataset))
        
        loaders, w_class, class_encoding = load_dataset(dataset)
        train_loader, val_loader, test_loader = loaders
        
        if args.mode.lower() in {'train'}:
            model,tl,tmiou,vl,vmiou = train(train_loader, val_loader, w_class, class_encoding)
            plt.plot(tl,label="train loss")
            plt.plot(tmiou,label="train miou")
            plt.plot(vl,label="val loss")
            plt.plot(vmiou,label="val miou")
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("loss/accuracy")
            plt.grid(True)
            plt.xticks()
            plt.savefig('./plots/train.png')
        elif args.mode.lower() == 'test':
            num_classes = len(class_encoding)
            #model = ENet(num_classes)
            model = ERFNet(num_classes)
            if use_cuda:
                model = model.cuda()
            optimizer = optim.Adam(model.parameters())
            model = utils.load_checkpoint(model, optimizer, args.save_dir, args.name)[0]
            test(model, test_loader, w_class, class_encoding)
        else:
            raise RuntimeError(
                "\"{0}\" is not a valid choice for execution mode.".format(
                    args.mode))
