#-*- coding:utf8 -*-
#!/usr/bin/env python
'''
@Author:qiuzhongxi
@Filename:seg_train.py
@Date:2020/3/16
@Software:PyCharm
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
from torch import distributed
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import json
import tqdm
import os
import glob
import copy

from torch.utils.tensorboard import SummaryWriter


from modules import unet
from loss_metrcs import iou_pytorch
from datasets import SegPathsDataset
from loss_metrcs import FocalLoss
from loss_metrcs import GDiceLoss
from loss_metrcs import Accuracy

with open("config.json") as f:
    config = json.load(f)

os.environ["CUDA_VISIBLE_DEVICES"] = config["gpu_index"]
def get_paths(img_dir, mask_dir, suffix=".png"):
    img_paths = glob.glob(os.path.join(img_dir, "*{}".format(suffix)))
    mask_paths = glob.glob(os.path.join(mask_dir,"*{}".format(suffix)))
    return img_paths,mask_paths

def updata_weights(model, state_dict):
    weights = {k:v for k,v in state_dict.items() if "out_conv" not in k}
    model_state = model.state_dict()
    model_state.update(weights)
    model.load_state_dict(model_state)
    return model

def one_hot(targets,num_classes):
    targets_extend=targets.clone()
    targets_extend.unsqueeze_(1) # convert to Nx1xHxW
    one_hot = torch.LongTensor(targets_extend.size(0), num_classes, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1)
    return one_hot

def main():
    epochs = config["epochs"]
    init_lr = config["init_lr"]
    momentum = config["momentum"]
    image_dir = config["image_dir"]
    mask_dir = config["mask_dir"]
    batch_size = config["batch_size"]
    suffix = config["suffix"]
    val_size = config["val_size"]
    num_classes = config["num_classes"]
    img_size = config["img_size"]
    log_dir = config["log_dir"]
    ckpt_dir = config["ckpt_dir"]
    model_name = config["model_name"]
    channels = config["channels"]
    alpha = 2
    gamma = 1
    num_workers = config["num_workers"]
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # train_image_dir = os.path.join(image_dir, "train_images")
    # train_mask_dir = os.path.join(mask_dir, "train_masks")
    # eval_image_dir = os.path.join(image_dir, "eval_images")
    # eval_mask_dir = os.path.join(mask_dir, "eval_masks")
    writer = SummaryWriter(log_dir=os.path.join(log_dir,model_name),comment=model_name)
    train_paths,train_label_paths = get_paths(image_dir, mask_dir, suffix)
    #val_paths, val_label_paths = get_paths(eval_image_dir, eval_mask_dir, suffix)
    train_paths, val_paths, train_label_paths, val_label_paths = train_test_split(train_paths, train_label_paths, 
    test_size=val_size, random_state=0)
    train_dataset = SegPathsDataset(train_paths, train_label_paths,img_size=img_size)
    val_datset = SegPathsDataset(val_paths, val_label_paths, augmentation=False,img_size=img_size)
    train_size, val_size = len(train_dataset), len(val_datset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print("train on {} samples, val on {} samples".format(train_size, val_size))
    val_loader = DataLoader(val_datset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = unet.Unet(channels, num_classes)
    #model = unet.ResUnet(channels,num_classes,18)
    #model = unet.MobileNetv3_Small_Unet(channels,num_classes)
    #model = unet.MobileNetv3_Large_Unet(channels,num_classes)
    # model = unet.ResUnetFPN(channels, num_classes, 18)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    if torch.cuda.device_count() > 1:
        print("Use {} GPUs".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            print("GPU name:{} Capability:{} state:{}".format(
                torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i),torch.cuda.get_device_properties(i)))
        model = torch.nn.parallel.DataParallel(model)

    model.to(device)
    #criterion = torch.nn.CrossEntropyLoss()
    # criterion = FocalLoss(alpha=0.25)
    ce = torch.nn.CrossEntropyLoss()
    #flloss = FocalLoss(alpha=alpha, gamma=gamma)
    dloss = GDiceLoss(num_classes)
    #optimizer = torch.optim.Adam(model.parameters(),init_lr,weight_decay=1e-2)
    optimizer = torch.optim.SGD(model.parameters(), init_lr, momentum=momentum)
    lr_scheduler=ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True,
        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    start_epoch = 0
    best_acc = 0
    best_mean_iou = 0
    if os.path.exists(os.path.join(ckpt_dir,"{}_last.pt".format(model_name))):
        state_dict = torch.load(os.path.join(ckpt_dir,"{}_last.pt".format(model_name)))
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        start_epoch = state_dict["epoch"]
        best_acc = state_dict["best_acc"]
        best_mean_iou = state_dict["best_mean_iou"]
        print("load last ckpt successful")

    last_state_dict = {
        "epoch": start_epoch,
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "best_acc": best_acc,
        "best_mean_iou": best_mean_iou
    }
    torch.save(last_state_dict, os.path.join(ckpt_dir, "{}_last.pt".format(model_name)))
    train_acc = Accuracy(num_classes)
    eval_acc = Accuracy(num_classes)
    try:
        for epoch in range(start_epoch,epochs):
            bar = tqdm.tqdm(train_loader)
            losses = 0.0
            total = 0
            flag = True
            ious = None
            ce_losses = 0.0
            # fl_losses = 0.0
            dl_losses = 0.0
            model.train()
            for x, label in bar:
                x = x.to(device,dtype=torch.float32)
                batch = x.size(0)
                label_type = torch.float32 if num_classes == 1 else torch.long
                label = label.to(device, dtype=label_type)
                optimizer.zero_grad()
                pred = model(x)
                #print(pred.shape)
                celoss = ce(pred, label)
                #focal_loss = flloss(pred, label)
                dice_loss = dloss(pred, label)
                loss = alpha*celoss + gamma*dice_loss
                loss.backward()
                optimizer.step()
                pred = F.log_softmax(pred, dim=1)
                _, pred = torch.max(pred, dim=1)
                losses += loss.item() * batch
                ce_losses += celoss.item() * batch
                # fl_losses += focal_loss.item() * batch
                dl_losses += dice_loss.item() * batch
                total += batch
                if flag:
                    ious = iou_pytorch(pred, label, averge=False)
                    flag = False
                else:
                    ious =torch.cat([ious,iou_pytorch(pred, label, averge=False)], dim=0)
                mean_iou = ious.mean().detach().cpu().numpy()
                pred = pred.detach().cpu().numpy()
                la = label.detach().cpu().numpy()
                train_acc.add_batch(pred, la)
                acc = train_acc.eval()
                acc_text = ""
                for k in acc.keys():
                    acc_text+=" {}:{:.3f}".format(k, acc[k])
                text = "{:03d}/{:03d} loss:{:.3f} ce_loss:{:.3f} dice_loss:{:.3f} mean_iou:{:.3f} acc:{}".format(
                        epoch+1, epochs, losses / total, ce_losses / total,  dl_losses / total,mean_iou, acc_text)
                bar.set_description(text)
            lr_scheduler.step(losses / total)
            acc = train_acc.eval()
            mean_acc=0
            for k in acc.keys():
                if "0" in k: continue
                mean_acc += acc[k]
            mean_acc /= (num_classes-1)
            writer.add_scalar("trian/mean_dice",mean_acc)
            writer.add("train/mean_iou",mean_iou.detach().cpu().numpy())
            train_acc.reset()
            with torch.no_grad():
                bar = tqdm.tqdm(val_loader)
                losses = 0.0
                total = 0
                flag = True
                ious = None
                last_mean_iou = 0
                model.eval()
                for x, label in bar:
                    x = x.to(device,dtype=torch.float32)
                    batch = x.size(0)
                    label_type = torch.float32 if num_classes == 1 else torch.long
                    label = label.to(device, dtype=label_type)

                    pred = model(x)
                    celoss = ce(pred, label)
                    # focal_loss = flloss(pred, label)
                    dice_loss = dloss(pred, label)
                    loss = alpha*celoss + gamma*dice_loss
                    pred = F.log_softmax(pred, dim=1)
                    _, pred = torch.max(pred, dim=1)

                    losses += loss.item() * batch
                    ce_losses += celoss.item() * batch
                    # fl_losses += focal_loss.item() * batch
                    dl_losses += dice_loss.item() * batch
                    if flag:
                        ious = iou_pytorch(pred, label, averge=False)
                        flag = False
                    else:
                        ious = torch.cat([ious, iou_pytorch(pred, label, averge=False)], dim=0)
                    total += batch
                    mean_iou = ious.mean().detach().cpu().numpy()
                    last_mean_iou = copy.deepcopy(mean_iou)
                    pred = pred.detach().cpu().numpy()
                    la = label.detach().cpu().numpy()
                    eval_acc.add_batch(pred, la)
                    acc = eval_acc.eval()
                    acc_text = ""
                    for k in acc.keys():
                        acc_text += " {}:{}".format(k,acc[k])
                    text = "{:03d}/{:03d} loss:{:.3f} ce_loss:{:.3f} dice_loss:{:.3f} mean_iou:{:.3f} acc:{}".format(
                        epoch + 1, epochs, losses / total, ce_losses / total, dl_losses / total,
                        mean_iou, acc_text)
                    bar.set_description(text)
                acc = eval_acc.eval()
                mean_acc = 0
                for k in acc.keys():
                    if acc[k] == -1: continue
                    mean_acc += acc[k]
                mean_acc /= (num_classes - 1)
                writer.add_scalar("eval/mean_dice",mean_acc)
                writer.add_scalar("eval/mean_iou",last_mean_iou)
                eval_acc.reset()
                if last_mean_iou > best_mean_iou:
                    best_mean_iou = last_mean_iou
                    best_state_dict_mean_iou = {
                        "epoch": epoch,
                        "model": copy.deepcopy(model.state_dict()),
                        "optimizer": copy.deepcopy(optimizer.state_dict()),
                        "best_acc": best_acc,
                        "best_mean_iou": best_mean_iou
                    }
                    torch.save(best_state_dict_mean_iou,os.path.join(ckpt_dir,"{}_best_iou.pt").format(model_name))
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_state_dict = {
                        "epoch":epoch,
                        "model":copy.deepcopy(model.state_dict()),
                        "optimizer":copy.deepcopy(optimizer.state_dict()),
                        "best_acc":best_acc,
                        "best_mean_iou":best_mean_iou
                    }
                    torch.save(best_state_dict,os.path.join(ckpt_dir,"{}_best.pt".format(model_name)))

            last_state_dict = {
                "epoch": epoch + 1,
                "model": copy.deepcopy(model.state_dict()),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "best_acc": best_acc,
                "best_mean_iou": best_mean_iou
            }
            torch.save(last_state_dict, os.path.join(ckpt_dir, "{}_last.pt".format(model_name)))

    except KeyboardInterrupt as e:
        print(e)


if __name__ == "__main__":
    main()