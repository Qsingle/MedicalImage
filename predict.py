#-*- coding:utf8 -*-
#!/usr/bin/env python
'''
@Author:qiuzhongxi
@Filename:predict.py
@Date:2020/3/17
@Software:PyCharm

Kits19 CT predict
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
from torch.nn import functional as F
import nibabel as nib
import os
import json
import tqdm
import numpy as np

from modules import unet

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

def main():
    '''
    The main function of predict.
    '''
    data_dir = config["data_dir"]
    ckpt_dir = config["ckpt_dir"]
    model_name = config["model_name"]
    num_classes = config["num_classes"]
    channels = config["channels"]
    output_dir = config["output_dir"]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    model = unet.Unet(channels, num_classes)
    #model = unet.ResUnet(channels, num_classes, 18)
    #model = unet.MobileNetv3_Large_Unet(channels, num_classes)
    #model = unet.MobileNetv3_Small_Unet(channels, num_classes)
    ckpt_path = os.path.join(ckpt_dir,"{}_last.pt".format(model_name))
    state_dict = torch.load(ckpt_path)
    try:
        model.load_state_dict(state_dict["model"])
    except RuntimeError:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict["model"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    bar = tqdm.tqdm(range(210, 300))
    model.eval()
    model.to(device)
    with torch.no_grad():
        for i in bar:
            image_dir = os.path.join(data_dir,"case_{:05d}".format(i))
            image_path = os.path.join(image_dir, "imaging.nii.gz")
            bar.set_description("predict {}".format(image_path))
            x_nib = nib.load(image_path)
            image = x_nib.get_data()
            image = normalize(image)
            outputs = []
            for index in range(image.shape[0]):
                img = image[index]
                if img.ndim == 2:
                    img = np.expand_dims(img, axis=0)
                img = torch.from_numpy(img)
                img = img.unsqueeze(0)
                img = img.to(device)
                img = img.float()
                pred = model(img)
                pred = F.log_softmax(pred, dim=1)
                _, pred = torch.max(pred, dim=1)
                pred = pred.squeeze(0)
                pred = pred.detach().cpu().numpy()
                outputs.append(pred)
            outputs = np.stack(outputs, axis=0)
            outputs = outputs.astype(np.uint8)
            outputs = np.clip(outputs, 0, 255)
            out_image = nib.Nifti1Image(outputs, x_nib.affine)
            nib.save(out_image, os.path.join(output_dir,"prediction_{:05d}.nii.gz".format(i)))
        print("predict successful")

if __name__ == '__main__':
    main()
