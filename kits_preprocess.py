#-*- coding:utf8 -*-
#!/usr/bin/env python
'''
@Author:qiuzhongxi
@Filename:kits_preprocess.py
@Date:2020/3/16
@Software:PyCharm

Preprocess the Kits19 CT data to npy
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import nibabel as nib
import os
import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--json_path",help="The path of the json file", required=True, type=str)
parser.add_argument("--data_dir", help="The directory of the dataset", required=True)
parser.add_argument("--output_dir", help="The output directory", default="./output")

args = parser.parse_args()

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

def kits2019_split(json_data, output_dir, data_dir):
    '''
    Split the kits 2019 data to 2d
    :param json_data:The kits json data, which contain the cids
    :param output_dir:The output directory
    :param data_dir:The directory of the data
    :return:None
    '''
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, "train_images"))
        os.mkdir(os.path.join(output_dir, "train_masks"))
        os.mkdir(os.path.join(output_dir,"eval_images"))
        os.mkdir(os.path.join(output_dir,"eval_masks"))
    train_case, eval_case = train_test_split(json_data, random_state=0, test_size=0.2)
    #split the training CT case to 2d npy data
    bar = tqdm.tqdm(train_case)
    for case in bar:
        case_id = case["case_id"]
        image_path = os.path.join(data_dir,case_id,"imaging.nii.gz")
        mask_path = os.path.join(data_dir, case_id,"segmentation.nii.gz")
        image = nib.load(image_path).get_data()
        image = normalize(image)
        label = nib.load(mask_path).get_data()
        for index in range(image.shape[0]):
            img = image[index]
            mask = label[index]
            mask = mask.astype(np.uint8)
            np.save(os.path.join(output_dir, "eval_images", "{}_{}.npy".format(case_id, index)), img)
            np.save(os.path.join(output_dir, "eval_masks", "{}_{}.npy".format(case_id, index)), mask)

    bar = tqdm.tqdm(eval_case)
    #split the eval CT case to 2d npy data
    for case in bar:
        case_id = case["case_id"]
        image_path = os.path.join(data_dir,case_id,"imaging.nii.gz")
        mask_path = os.path.join(data_dir, case_id,"segmentation.nii.gz")
        image = nib.load(image_path).get_data()
        image = normalize(image)
        label = nib.load(mask_path).get_data()
        for index in range(image.shape[0]):
            img = image[index]
            img = img
            mask = label[index]
            mask = mask.astype(np.uint8)
            np.save(os.path.join(output_dir,"eval_images","{}_{}.npy".format(case_id, index)), img)
            np.save(os.path.join(output_dir, "eval_masks", "{}_{}.npy".format(case_id, index)), mask)

    with open("./train.txt",encoding="utf-8",mode="w") as f:
        for case in train_case:
            line = case["case_id"] + "\n"
            f.write(line)
    with open("./eval.txt", encoding="utf-8", mode="w") as f:
        for case in eval_case:
            line = case["case_id"] + "\n"
            f.write(line)

def main():
    json_path = args.json_path
    data_dir = args.data_dir
    output_dir = args.output_dir
    with open(json_path) as f:
        data = json.load(f)
        kits2019_split(data, output_dir, data_dir)

if __name__ == "__main__":
    main()
