#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 01:41:46 2022

@author: sen
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 21 11:48:17 2022

@author: sen
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from segnet_model import SEGNET
from camvid_dataloader import CamVidDataset
from camvid_utils import iou, pixel_acc
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from torchvision import models
from torchvision.models.vgg import VGG

nworkers = 2
nclasses = 32
batch_size = 8
epochs = 100
lr = 0.0002
weight_decay = 0.0005
device = "cuda" if torch.cuda.is_available() else "cpu"
accum_iter = 8  

root_dir   = "CamVid/"
train_file = os.path.join(root_dir, "train.txt")
val_file   = os.path.join(root_dir, "val.txt")
path_cpt_file = 'cpts/segnet.cpt'
save_model = True

def train (train_loader, model, optimizer, loss_f):
    loop = tqdm(train_loader, leave = True)
    model.train()
    total_ious = []
    pixel_accs = []
    for batch_idx, batch in enumerate(loop):
        x, y = Variable(batch['X']).to(device), Variable(batch['Y']).to(device)
        out = model(x)
        N, _, h, w = out.shape
        pred = out.data.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, nclasses).argmax(axis=1).reshape(N, h, w)
        del x
        loss_val = loss_f(out, y)
        target = batch['l'].cpu().numpy().reshape(N, h, w)
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            pixel_accs.append(pixel_acc(p, t))
        del y
        del out
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss = loss_val.item())


    total_ious = np.array(total_ious).T  
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
        
    return (float(loss_val.item()), np.nanmean(ious), pixel_accs)

def test (test_loader, model, loss_f):
    loop = tqdm(test_loader, leave = True)
    model.eval()
    total_ious = []
    pixel_accs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loop):
            x, y = Variable(batch['X']).to(device), Variable(batch['Y']).to(device)
            out = model(x)
            N, _, h, w = out.shape
            pred = out.data.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, nclasses).argmax(axis=1).reshape(N, h, w)
            del x
            loss_val = loss_f(out, y)
            target = batch['l'].cpu().numpy().reshape(N, h, w)
            for p, t in zip(pred, target):
                total_ious.append(iou(p, t))
                pixel_accs.append(pixel_acc(p, t))
            del y
            del out
            # update progress bar
            loop.set_postfix(loss = loss_val.item())
        
    total_ious = np.array(total_ious).T  
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
        
    return (float(loss_val.item()), np.nanmean(ious), pixel_accs)


def main():
    vgg_model = models.vgg16(pretrained=True)
    segnet_model = SEGNET(pretrained=vgg_model, nclasses=nclasses).to(device)
    optimizer = optim.Adam(segnet_model.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, 
                                                                T_mult=2,
                                                                eta_min=0.000002,
                                                                last_epoch=-1)
    loss_f = nn.BCEWithLogitsLoss()
    

    train_dataset = CamVidDataset(file = train_file, phase ='train')
    
    test_dataset = CamVidDataset(file = val_file, phase ='val')
    
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, 
                              num_workers = nworkers, shuffle = True)
    
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, 
                              num_workers = nworkers, shuffle = True)
    
    loss_lst = []
    iou_avg = [] 
    pixel_avg_acc = [] 

    loss_lst_test = []
    iou_avg_test = [] 
    pixel_avg_acc_test = []

    for epoch in range(epochs):
        loss_value, iou_avg_val, pixel_avg_val  = train(train_loader, segnet_model, optimizer, loss_f)
        loss_lst.append(loss_value)
        iou_avg.append(iou_avg_val)
        pixel_avg_acc.append(pixel_avg_val)

        loss_value_test, iou_avg_val_test, pixel_avg_val_test  = test(test_loader, segnet_model, loss_f)
        loss_lst_test.append(loss_value_test)
        iou_avg_test.append(iou_avg_val_test)
        pixel_avg_acc_test.append(pixel_avg_val_test)
        print(f"Epoch:{epoch}  Train[Loss:{loss_value}  Avg_IoU:{iou_avg[-1]}  Pix_acc:{pixel_avg_acc[-1]}]")
        print(f"Epoch:{epoch}  Test[Loss:{loss_value_test}  Avg_IoU:{iou_avg_test[-1]}  Pix_acc:{pixel_avg_acc_test[-1]}]")
        scheduler.step()
 
        if epoch == epochs - 1: 
            with open('results/segnet_loss.txt','w') as values:
                values.write(str(loss_lst))
            with open('results/segnet_iou.txt','w') as values:
                values.write(str(iou_avg))
            with open('results/segnet_pixel_acc.txt','w') as values:
                values.write(str(pixel_avg_acc))
            with open('results/segnet_loss_test.txt','w') as values:
                values.write(str(loss_lst_test))
            with open('results/segnet_iou_test.txt','w') as values:
                values.write(str(iou_avg_test))
            with open('results/segnet_pixel_acc_test.txt','w') as values:
                values.write(str(pixel_avg_acc_test))
            torch.save({
                'model_state_dict': segnet_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, path_cpt_file)
            save_model = False
            print("Results and Model Stores!")
            break
        
         
        
        
if __name__ == "__main__":
    main()

