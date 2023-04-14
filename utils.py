'''
Implementation of CutMix data augmentation technique and MixUp data augmentation technique.
https://github.com/hysts/pytorch_cutmix/blob/master/cutmix.py
https://github.com/hysts/pytorch_mixup/blob/master/utils.py

We tried to use the code below but had several issues while training. 
Due to timing constraints, we decided to disable these function calls for now.
The original ConvMixer is trained using the timm library which contains these augmentations by default
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cutmix(batch, alpha):
    data, targets = batch

    # generate mixed sample
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    # calculate bbox
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    # paste shuffled image on top of original image
    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets = (targets, shuffled_targets, lam)

    return data, targets 

# one hot encoding function
def onehot(label, n_classes):
    return torch.zeros(label.size(0), n_classes).scatter_(
        1, label.view(-1, 1), 1)


def mixup(data, targets, alpha, n_classes):
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    targets = onehot(targets, n_classes)
    targets2 = onehot(targets2, n_classes)

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    # create liear combination of two random samples
    data = data * lam + data2 * (1 - lam)
    targets = (targets, targets2, lam)
    return data, targets

# randomly apply either cutmix or mixup
# requires both cutmix and mixup flags to be enabled
class CustomCollator:
    def __init__(self, cutmix_alpha, mixup_alpha, num_classes):
        self.cutmix_alpha = cutmix_alpha
        self.mixup_alpha = mixup_alpha
        self.num_classes = num_classes

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        if np.random.uniform() < 0.5:
            batch = cutmix(batch, self.cutmix_alpha)
        else:
            batch = mixup(*batch, self.mixup_alpha, self.num_classes)
        return batch
    
# deprecated

# class CutMixCriterion:
#     def __init__(self, reduction):
#         self.criterion = nn.CrossEntropyLoss(reduction=reduction)

#     def __call__(self, preds, targets):
#         targets1, targets2, lam = targets
#         return lam * self.criterion(
#             preds, targets1) + (1 - lam) * self.criterion(preds, targets2)
            

# def cross_entropy_loss(input, target, size_average=True):
#     input = F.log_softmax(input, dim=1)
#     loss = -torch.sum(input * target)
#     if size_average:
#         return loss / input.size(0)
#     else:
#         return loss


# class CrossEntropyLoss(object):
#     def __init__(self, size_average=True):
#         self.size_average = size_average

#     def __call__(self, input, target):
#         return cross_entropy_loss(input, target, self.size_average)
