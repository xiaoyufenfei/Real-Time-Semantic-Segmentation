import torch
import cv2
import torch.utils.data
import scipy.misc as m

__author__ = "Sachin Mehta"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Sachin Mehta"

import numpy as np
from cityscape import cityscapes_labels
def replace_city_labels(label_data):
    labels = cityscapes_labels.labels
    converted = np.ones(label_data.shape, dtype=np.float) * 255
    # id to trainId
    id2trainId = {label.id: label.trainId for label in labels}
    for id in id2trainId:
        trainId = id2trainId[id]
        converted[label_data == id] = trainId
    return converted
class MyDataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, imList, labelList, transform=None, data_name = 'cityscape'):
        '''
        :param imList: image list (Note that these lists have been processed and pickled using the loadData.py)
        :param labelList: label list (Note that these lists have been processed and pickled using the loadData.py)
        :param transform: Type of transformation. SEe Transforms.py for supported transformations
        '''
        self.imList = imList
        self.labelList = labelList
        self.transform = transform
        self.data_name = data_name

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        '''

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        image_name = self.imList[idx]
        label_name = self.labelList[idx]
        image = cv2.imread(image_name)
        if self.data_name == 'cityscape':
            label = cv2.imread(label_name, 0)
            label = replace_city_labels(label)
            label[label == 255] = 19
        elif self.data_name == 'camVid':
            label = m.imread(label_name)
            label = np.array(label,dtype='float32')
        if self.transform:
            [image, label] = self.transform(image, label)
        return (image, label)
