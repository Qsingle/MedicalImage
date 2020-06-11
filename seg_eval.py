#-*- coding:utf8 -*-
#!/usr/bin/env python
'''
@Author:qiuzhongxi
@Filename:seg_eval.py
@Date:2020/3/18
@Software:PyCharm
'''
import numpy as np
import torch
from torch.nn import functional as F
import os
import nibabel as nib
import json

from modules import unet
from loss_metrcs import iou_pytorch


with open("config.json") as f:
    config = json.load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_index"]

def normalize(vol):
   '''
    Min-Max normalize the case
    vol: ndarray, CT volumn
    return: ndarray, normalized CT volumn
    '''
    hu_max = 512
    hu_min = -512
    vol = np.clip(vol, hu_min, hu_max)

    mxval = np.max(vol)
    mnval = np.min(vol)
    volume_norm = (vol - mnval) / max(mxval - mnval, 1e-3)
    return volume_norm
def dice(ground_truth:np.array, predictions, case_id):
    '''
    Calculate the dice for one CT case.
    '''
    if len(predictions.shape) == 4:
        predictions = np.argmax(predictions, axis=-1)

    if not isinstance(predictions, (np.ndarray, nib.Nifti1Image)):
        raise ValueError("Predictions must by a numpy array or Nifti1Image")
    if isinstance(predictions, nib.Nifti1Image):
        predictions = predictions.get_data()

    if not np.issubdtype(predictions.dtype, np.integer):
        predictions = np.round(predictions)
    predictions = predictions.astype(np.uint8)

    if not predictions.shape == ground_truth.shape:
        raise ValueError(
            ("Predictions for case {} have shape {} "
             "which do not match ground truth shape of {}").format(
                case_id, predictions.shape, ground_truth.shape
            )
        )

    try:
        # Compute tumor+kidney Dice
        tk_pd = np.greater(predictions, 0)
        tk_gt = np.greater(ground_truth, 0)
        tk_dice = 2 * np.logical_and(tk_pd, tk_gt).sum() / (
                tk_pd.sum() + tk_gt.sum()
        )
    except ZeroDivisionError:
        return 0.0, 0.0

    try:
        # Compute tumor Dice
        tu_pd = np.greater(predictions, 1)
        tu_gt = np.greater(ground_truth, 1)
        tu_dice = 2 * np.logical_and(tu_pd, tu_gt).sum() / (
                tu_pd.sum() + tu_gt.sum()
        )
    except ZeroDivisionError:
        return tk_dice, 0.0

    return tk_dice, tu_dice


def iou_numpy(outputs: np.array, labels: np.array):
    #outputs = outputs.squeeze(1)
    SMOOTH = 1e-9
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return thresholded  # Or thresholded.mean()

def eval():
    data_dir = config["data_dir"]
    channels = config["channels"]
    ckpt_dir = config["ckpt_dir"]
    num_classes = config["num_classes"]
    model_name = config["model_name"]
    txt_dir = os.path.abspath(os.path.dirname(os.path.dirname(data_dir)))
    txt_path = os.path.join(txt_dir,"eval.txt")
    with open(txt_path,"r") as f:
        lines = f.readlines()
    # model = unet.Unet(channels, num_classes)
    # model = unet.ResUnet(channels,num_classes,18)
    # model = unet.MobileNetv3_Small_Unet(channels,num_classes)
    # model = unet.MobileNetv3_Large_Unet(channels,num_classes)
    model = unet.ResUnetFPN(channels, num_classes, 18)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    state_dict = torch.load(os.path.join(ckpt_dir,"{}_last.pt".format(model_name)),map_location="cpu")
    model.load_state_dict(state_dict["model"])
    model.to(device)
    total_ground_truths = []
    total_preds = []
    for line in lines:
        line=line.strip()
        case_dir = os.path.join(data_dir,line)
        image_path = os.path.abspath(os.path.join(case_dir,"imaging.nii.gz"))
        label_path = os.path.abspath(os.path.join(case_dir,"segmentation.nii.gz"))
        image = nib.load(image_path).get_data()
        label = nib.load(label_path).get_data()
        image = normalize(image)
        preds = []
        label = label.astype(np.uint8)
        for index in range(image.shape[0]):
            img = image[index]
            if img.ndim == 2:
                img = np.expand_dims(img, 0)
            img = torch.from_numpy(img)
            img=img.unsqueeze(0)
            img=img.to(device,dtype=torch.float32)
            pred = model(img)
            pred = F.log_softmax(pred, dim=1)
            _, pred = torch.max(pred, dim=1)
            pred = pred.squeeze(0)
            pred = pred.detach().cpu().numpy()
            preds.append(pred)
            total_ground_truths.append(label[index])
            total_preds.append(pred.astype(np.uint8))
        preds = np.stack(preds,axis=0)
        preds = preds.astype(np.uint8)
        tk_dice, tu_dicet = dice(label, preds,line)
        tk_iou = tk_dice/(2-tk_dice)
        tu_iou = tu_dicet/(2-tu_dicet)
        print("Case:{} Dice of kidney and tumor:{} Dice of tumor:{} kidney iou:{} tumor iou:{}".format(line,tk_dice,tu_dicet,tk_iou,tu_iou))
    total_ground_truths = np.stack(total_ground_truths, axis=0).astype(np.uint8)
    total_preds = np.stack(total_preds, axis=0).astype(np.uint8)
    tk_dice, tu_dicet = dice(total_ground_truths, total_preds, 0)
    tk_iou = tk_dice / (2 - tk_dice)
    tu_iou = tu_dicet / (2 - tu_dicet)
    print("Total Dice of kidney and tumor:{} Dice of tumor:{}, iou of kidney:{} iou of tumor:{}".format(tk_dice, tu_dicet,tk_iou, tu_iou))
    print("kidney iou:{} ")

if __name__ == "__main__":
    eval()