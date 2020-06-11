#-*- coding:utf8 -*-
#!/usr/bin/env python
'''
@Author:qiuzhongxi
@Filename:datasets.py
@Date:2020/3/7
@Software:PyCharm

Some Dataset Class for this project
'''
from torch.utils.data import Dataset
import torch
import cv2
from PIL import Image
import numpy as np
import os
from matplotlib import pyplot as plt
from albumentations import Compose
from albumentations import HorizontalFlip,VerticalFlip,RandomGamma,RandomBrightnessContrast,PadIfNeeded,ShiftScaleRotate
from albumentations import Normalize

class FolderDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super(FolderDataset, self).__init__()
        assert os.path.exists(data_dir), "The directory {} not exists".format(data_dir)
        self.paths, self.ids = self.get_paths(data_dir)
        self.transform = transform

    def process(self, img, img_size=224):
        img = img.resize((img_size, img_size))
        img = np.asarray(img)
        if img.ndim > 2:
            img = np.transpose(img, [2, 1, 0])
        return img

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.ids[index]
        img = Image.open(path)
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
            return img, label
        img = self.process(img)
        return img, label

    def __len__(self):
        return len(self.paths)

    def show_batch(self, rows=5, cols=None):
        if cols is None:
            cols = rows
        total = 5 * 5
        font_dict = {'fontsize': 7,
                     'fontweight': 2,
                     'verticalalignment': 'baseline',
                     'horizontalalignment': "center"}
        plt.figure(dpi=224)
        for i in range(total):
            random = np.random.randint(0, self.__len__())
            plt.subplot(rows, cols, i + 1)
            plt.title(self.classes[self.ids[random]], fontdict=font_dict, pad=1.2)
            img = Image.open(self.paths[random])
            img = img.resize((96, 96))
            img = np.asarray(img)
            if img.ndim <= 2:
                plt.imshow(img, cmap="gray")
            else:
                plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
        plt.show()

    def statistic(self):
        counters = []
        for unique in np.unique(self.ids):
            counters.append(np.sum(unique == self.ids))

        plt.figure(dpi=224)
        plt.bar(range(len(counters)), counters)
        #plt.show()
        plt.savefig("out.png")

    def get_paths(self, data_dir):
        paths = []
        ids = []
        class_dict = dict()
        classes = []
        cl = 0
        for home, dirs, _ in os.walk(data_dir):
            for dir in dirs:
                if dir not in class_dict:
                    classes.append(dir)
                    class_dict[dir] = cl
                    cl += 1
                img_dir = os.listdir(os.path.join(home, dir))
                for path in img_dir:
                    if path.endswith("jpg") or path.endswith("png") or path.endswith("jpeg"):
                        paths.append(os.path.join(home, dir, path))
                        id = class_dict[dir]
                        ids.append(id)
        self.class_dict = class_dict
        self.classes = classes
        return paths, ids

class SegPathsDataset(Dataset):
    def __init__(self, image_paths, label_paths, augmentation=True,img_size=256):
        super(SegPathsDataset,self).__init__()
        assert len(image_paths) == len(label_paths), "The length is not equal, len(image_paths)/len(label_paths)={}/" \
                                                     "{}".format(len(image_paths), len(label_paths))
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.augmentation = augmentation
        self.length = len(image_paths)
        self.img_size = img_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        mask_path = self.label_paths[index]
        if img_path.endswith(".npy"):
            img = np.load(img_path)
        else:
            img = cv2.imread(img_path)
        if mask_path.endswith(".npy"):
            mask = np.load(mask_path)
        else:
            mask = cv2.imread(mask_path, 0)
        if self.augmentation:
            task = [
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                RandomGamma(),
                RandomBrightnessContrast(p=0.5),
                PadIfNeeded(self.img_size,self.img_size),
                ShiftScaleRotate(scale_limit=0.5, p=0.5),
                #Normalize(mean=[0.210, 0.210, 0.210], std=[0.196, 0.196, 0.196], always_apply=True)
            ]
            aug = Compose(task)
            aug_data = aug(image=img, mask=mask)
            img, mask = aug_data["image"], aug_data["mask"]
        img = self._normalize(img)
        img = cv2.resize(img,(self.img_size,self.img_size))
        mask = cv2.resize(mask,(self.img_size,self.img_size))
        mask = mask // 255.0
        if img.ndim < 3:
            img = np.expand_dims(img, 0)
        else:
            img = np.transpose(img, axes=[2, 0, 1])
        return torch.from_numpy(img), torch.from_numpy(mask)
    
    def _normalize(self, img):
        normal_img = np.clip(img, 0, 255)
        maxval = np.max(img)
        minval = np.min(img)        
        normal_img = (img - minval) / max( maxval- minval, 1e-3)
        return normal_img

class PathsDataset(Dataset):
    def __init__(self, paths:list, data_dir, classes_dict=None, ids=None,augumentation=False,img_size=224,transform=None,suffix=".png"):
        super(PathsDataset, self).__init__()
        self.filename = paths
        self.data_dir = data_dir
        self.ids = ids
        self.transform = transform
        self.length = len(self.filename)
        self.classes_dict = classes_dict
        self.augumentation = augumentation
        self.img_size = img_size
        if suffix.startswith("."):
            self.suffix = suffix
        else:
            self.suffix = ".{}".format(suffix)

    def show_batch(self, rows=5, cols=None):
        if cols is None:
            cols = rows
        font_dict = {'fontsize': 7,
                     'fontweight': 2,
                     'verticalalignment': 'baseline',
                     'horizontalalignment': "center"}
        plt.figure(dpi=224)
        for i in range(cols*rows):
            random = np.random.randint(0, self.length-1)
            path = os.path.join(self.data_dir, self.filename[random]+self.suffix)
            img = Image.open(path)
            img = np.asarray(img)
            plt.subplot(cols, rows, i+1)
            text = self.classes_dict[self.ids[random]]
            plt.title(text, font_dict, pad=1.2)
            if img.ndim <= 2:
                plt.imshow(img,cmap="gray")
            else:
                plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
        plt.show()

    def __getitem__(self, index):
        path = os.path.join(self.data_dir, self.filename[index]+self.suffix)
        label = None if self.ids is None else self.ids[index]
        img = Image.open(path)
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        else:
            if self.augumentation:
                img = np.asarray(img)
                task = [
                    HorizontalFlip(p=0.5),
                    VerticalFlip(p=0.5),
                    RandomGamma(),
                    RandomBrightnessContrast(p=0.5),
                    PadIfNeeded(self.img_size,self.img_size),
                    ShiftScaleRotate(scale_limit=0.5, p=0.5)
                ]
                aug = Compose(task)
                aug_data = aug(image=img)
                img = aug_data["image"]
                #img = cv2.resize(img,(self.img_size,self.img_size))
                img = Image.fromarray(img)
            img = self.transform(img)
        if label is not None:
            return img,label
        else:
            return img

    def __len__(self):
        return self.length

if __name__ == "__main__":
    dataset = FolderDataset("../../data/256_ObjectCategories/")
    dataset.show_batch()