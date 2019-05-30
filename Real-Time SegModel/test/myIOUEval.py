import torch
import numpy as np

#adapted from https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/score.py

class iouEval:
    def __init__(self, nClasses, data_length,ignore_label=[]):
        self.nClasses = nClasses
        self.data_length = data_length
        self.ignore_label = ignore_label
        self.label = range(nClasses)
        self.reset()

    def reset(self):
        self.overall_acc = 0
        self.per_class_acc = np.zeros(self.nClasses, dtype=np.float32)
        self.per_class_iu = np.zeros(self.nClasses, dtype=np.float32)
        self.mIOU = 0
        self.batchCount = 1
        self.hist = np.zeros([self.nClasses,self.nClasses])

    def fast_hist(self, a, b):
        k = (a >= 0) & (a < self.nClasses)
        return np.bincount(self.nClasses * a[k].astype(int) + b[k], minlength=self.nClasses ** 2).reshape(self.nClasses, self.nClasses)

    def compute_hist(self, predict, gth):
        hist = self.fast_hist(gth, predict)
        return hist

    def addBatch(self, predict, gth):
        if type(predict).__module__ == np.__name__:
            predict = predict.flatten()
            gth = gth.flatten()
        else:
            predict = predict.cpu().numpy().flatten()
            gth = gth.cpu().numpy().flatten()

        self.hist += self.compute_hist(predict, gth)
        # overall_acc = np.diag(hist).sum() / (hist.sum() + epsilon)
        # per_class_acc = np.diag(hist) / (hist.sum(1) + epsilon)
        # per_class_iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)
        # mIou = np.nanmean(per_class_iu)
        # #print mIou
        # if self.hist == None:
        #     self.hist = hist
        # else:
        #     self.hist += hist
        # self.overall_acc +=overall_acc
        # self.per_class_acc += per_class_acc
        # self.per_class_iu += per_class_iu
        # self.mIOU += mIou
    def caculate_per_class_iu(self):
        class_iu = np.zeros(self.nClasses)
        for i in self.label:

            if  i in self.ignore_label:
                class_iu[i] = float('nan')
            else:
                tp = self.hist[i,i]
                fn = self.hist[i,:].sum() - tp
                notIgnored = [l for l in self.label if not l in self.ignore_label and l!=i]
                fp = self.hist[notIgnored, i].sum()
                denom = (tp + fp + fn)
                if denom == 0:
                    class_iu[i] = float('nan')
                else:
                    class_iu[i] = float(tp) / denom
        return class_iu
    def getMetric(self):
        epsilon = 0.0000001
        #overall accuracy
        overall_acc = np.diag(self.hist).sum() / (self.hist.sum() + epsilon)
        per_class_acc = np.diag(self.hist) / (self.hist.sum(1) + epsilon)
        per_class_iu = self.caculate_per_class_iu()
        # per_class_iu = np.diag(self.hist) / (self.hist.sum(1) + self.hist.sum(0) - np.diag(self.hist) + epsilon)
        mIOU = np.nanmean(per_class_iu)

        return overall_acc, per_class_acc, per_class_iu, mIOU