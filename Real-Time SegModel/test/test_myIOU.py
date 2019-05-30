import os
import glob
import scipy.misc as m
import PIL.Image as Image
import numpy as np
from myIOUEval import iouEval
import tqdm
def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    '''
    img[img == 255] = 19
    img[img == 33] = 18
    img[img == 32] = 17
    img[img == 31] = 16
    img[img == 28] = 15
    img[img == 27] = 14
    img[img == 26] = 13
    img[img == 25] = 12
    img[img == 24] = 11
    img[img == 23] = 10
    img[img == 22] = 9
    img[img == 21] = 8
    img[img == 20] = 7
    img[img == 19] = 6
    img[img == 17] = 5
    img[img == 13] = 4
    img[img == 12] = 3
    img[img == 11] = 2
    img[img == 8] = 1
    img[img == 7] = 0
    img[img == 0] = 255
    return img
if __name__ == '__main__':
    CITYSCAPES_RESULTS = '/home/zhengxiawu/work/real_time_seg/result/Espnet_cityscape'
    CITYSCAPES_DATASET = '/home/zhengxiawu/data/cityscapes/gtFine_trainvaltest'
    groundTruthSearch = os.path.join(CITYSCAPES_DATASET, "gtFine", "val", "*", "*_gtFine_labelIds.png")
    groundTruthImgList = glob.glob(groundTruthSearch)
    ignore_label = [0,1,2,3,4,5,6,9,10,14,15,16,18,29,30]
    iouEvalVal = iouEval(34, 500,ignore_label=ignore_label)

    for i in tqdm.tqdm(groundTruthImgList):
        prediction_file_name = os.path.join(CITYSCAPES_RESULTS,i.split('/')[-1].replace('gtFine_labelIds','leftImg8bit'))
        assert os.path.isfile(prediction_file_name),'not a file!!'
        predictionImg = Image.open(prediction_file_name)
        predictionNp = np.array(predictionImg)
        #predictionNp = relabel(predictionNp)
        groundTruthImg = Image.open(i)
        groundTruthNp = np.array(groundTruthImg)
        #groundTruthNp = relabel(predictionNp)
        iouEvalVal.addBatch(predictionNp, groundTruthNp)
    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalVal.getMetric()
    print per_class_iu
    print mIOU

    pass