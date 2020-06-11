#-*- coding:utf8 -*-
#!/usr/bin/env python
'''
@Author:qiuzhongxi
@Filename:loss_metrcs.py
@Date:2020/1/31
@Software:PyCharm
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from distutils.version import LooseVersion
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self,num_classes):
        super(MulticlassDiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, output, target, weights=None, ignore_index=None):
        """
        :param
        output: NxCxHxW
        Variable
        :param
        target: NxHxW
        LongTensor
        :param
        weights: C
        FloatTensor
        :param
        ignore_index: int
        index
        to
        ignore
        from loss
        :param
        binary: bool
        for binarized one chaneel(C=1) input
        :return:
        """
        output = F.softmax(output, dim=1)
        eps = 0.0001
        encoded_target = output.detach() * 0

        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)

class GDiceLoss(nn.Module):
    '''
    Generalized DiceLoss
    '''
    def __init__(self, num_classes):
        super(GDiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self,  output, target, weights=None, ignore_index=None):
        output = F.softmax(output, dim=1)
        eps = 0.0001
        encoded_target = output.detach() * 0
        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
        w = torch.sum(encoded_target, dim=(0, 2, 3))
        #print(w.shape)
        w = 1 / (w**2 + eps)
        intersection = output * encoded_target
        numerator = w * intersection.sum(0).sum(1).sum(1)
        denominator = (output + encoded_target)

        if weights is None:
            weights = 1

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = w*denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (2 * numerator / denominator))

        return loss_per_channel.sum() / output.size(1)



class FocalLoss(nn.Module):
    '''
    FocalLoss for multi class
    '''
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255, average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=weight,ignore_index=ignore_index)
        self.average = average

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        if self.average:
            return loss.mean()
        else:
            return loss.sum()


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss

def iou(pred, label,n_class):
    ious = []
    pred = pred.view(-1)
    label = label.view(-1)
    for cls in range(1,n_class):
        pred_ids = pred == cls
        label_ids = label == cls
        intersection = torch.sum(pred_ids * label_ids)
        union = pred_ids.sum() + label_ids.sum() - intersection

        iou = intersection.float() / union.float()
        if union.eq(0):
            ious.append(float('nan'))
        else:
            ious.append(iou.detach().cpu().float().numpy())

    return np.array(ious)


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, thr=0.5, averge=True):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    SMOOTH = 1e-6
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - thr), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    if averge:
        return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch
    else:
        return thresholded

class Accuracy:
    '''
    Eval the model with dice.
    '''
    def __init__(self, num_classes):
        np.seterr(divide='ignore', invalid='ignore')
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_martrix = np.zeros((self.num_classes, self.num_classes))
        self.dc_per_case = np.zeros(self.num_classes)

    def _generate_matrix(self, pred, label):
        mask = (label > 0) & (label < self.num_classes)
        label = self.num_classes * label[mask].astype('int') + pred[mask]
        count = np.bincount(label, minlength=self.num_classes**2)
        confusion_matrix = count.reshape(self.num_classes,self.num_classes)
        return confusion_matrix

    def dice_coef(self, class_, confusion_matrix):
        dc = confusion_matrix[class_][class_] * 2 / (
            np.sum(confusion_matrix, axis=0)[class_] + np.sum(confusion_matrix, axis=1)[class_]
        )
        if np.isnan(dc):
            dc = -1
        return dc

    def add_batch(self, preds:np.array, labels:np.array):
        assert preds.shape == labels.shape, "The shape of preds if not equal to the shape of labels, {}/{}" \
                                            .format(preds.shape,labels.shape)
        for i in range(len(preds)):
            self.add(preds[i], labels[i])

    def add(self, pred:np.array, label:np.array):
        assert pred.shape == label.shape, "The shape of pred is not equal to the shape of label, {}/{}".format(
            pred.shape, label.shape
        )
        matrix = self._generate_matrix(pred,label)
        self.confusion_martrix += matrix


    def eval(self):
        acc = dict()
        for cls in range(self.num_classes):
            dc_global = self.dice_coef(cls, self.confusion_martrix)
            acc[f'dc_global_{cls}'] = dc_global
        return acc

class TverskyLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, true, logits, eps=1e-7):
        """Computes the Tversky loss [1].
            Args:
                true: a tensor of shape [B, H, W] or [B, 1, H, W].
                logits: a tensor of shape [B, C, H, W]. Corresponds to
                    the raw output or logits of the model.
                alpha: controls the penalty for false positives.
                beta: controls the penalty for false negatives.
                eps: added to the denominator for numerical stability.
            Returns:
                tversky_loss: the Tversky loss.
            Notes:
                alpha = beta = 0.5 => dice coeff
                alpha = beta = 1 => tanimoto coeff
                alpha + beta = 1 => F beta coeff
            References:
                [1]: https://arxiv.org/abs/1706.05721
            """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        fps = torch.sum(probas * (1 - true_1_hot), dims)
        fns = torch.sum((1 - probas) * true_1_hot, dims)
        num = intersection
        denom = intersection + (self.alpha * fps) + (self.beta * fns)
        tversky_loss = (num / (denom + eps)).mean()
        return (1 - tversky_loss)

if __name__ == "__main__":
    pred = [[0,1,2,3,4],
            [0,1,2,3,4]]
    label = [[0,0,2,3,4],
             [0,1,2,3,4]]
    pred = np.array(pred)
    label = np.array(label)
    acc = Accuracy(5)
    print(pred.shape)
    acc.add_batch(pred,label)
    for i in range(label.shape[0]):
        print(acc.eval())
    Gdl = GDiceLoss(3)
    pred = np.random.normal(0, 1, [1, 3, 64, 64])
    label = np.random.randint(0, 3, [1, 64, 64])
    pred = torch.from_numpy(pred)
    label = torch.from_numpy(label).long()
    Gdl.forward(pred, label)