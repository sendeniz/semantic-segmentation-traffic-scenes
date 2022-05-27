#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 15:16:07 2022

@author: sen
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
test_model = False
class SEGNET(nn.Module):
    def __init__(self, pretrained, nclasses):    
        super().__init__()
        self.nclasses = nclasses
        self.pretrained = pretrained
        
        # Encoder body
        # Block 1
        self.block1 = nn.Sequential(
             nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1),
             nn.BatchNorm2d(64),
             nn.ReLU(inplace=True),
             nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
             nn.BatchNorm2d(64),  
             nn.ReLU(inplace=True)
            )
        
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            )
        
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
            )
        

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )

        # Block 4
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            )
        
        # Block 5 
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            )
        
        self.init_vgg_weights()
        
        # Decoder body
        # Block 5b
        self.block5b = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            )
        
        # Block 4b
        self.block4b = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )
        
        # Block 3b
        self.block3b = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            )
        
        # Block 2b
        self.block2b = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            )
        
        # Block 1b
        self.block1b = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            )
        self.classifier = nn.Sequential(
            nn.Conv2d(64, nclasses, kernel_size=(3,3), padding=1)
            )        
        
    def forward(self, x):
        # Stage 1 
        x = self.block1(x)
        x1_size = x.size()
        x, x1_maxpool_indices = self.maxpool(x)
        
        # Stage 2
        x = self.block2(x)
        x2_size = x.size()
        x, x2_maxpool_indices = self.maxpool(x)
        
        # Stage 3   
        x = self.block3(x)
        x3_size = x.size()
        x, x3_maxpool_indices = self.maxpool(x)
        
        # Stage 4
        x = self.block4(x)
        x4_size = x.size()
        x, x4_maxpool_indices = self.maxpool(x)
        
        # Stage 5
        x = self.block5(x)
        x5_size = x.size()
        x, x5_maxpool_indices = self.maxpool(x)
        
        # Stage 5d
        x = F.max_unpool2d(x, x5_maxpool_indices, kernel_size=2, stride=2, output_size=x5_size)
        x = self.block5b(x)
        
        # Stage 4d
        x = F.max_unpool2d(x, x4_maxpool_indices, kernel_size=2, stride=2, output_size=x4_size)
        x = self.block4b(x)

        # Stage 3d
        x = F.max_unpool2d(x, x3_maxpool_indices, kernel_size=2, stride=2, output_size=x3_size)
        x = self.block3b(x)
        
        # Stage 2d
        x = F.max_unpool2d(x, x2_maxpool_indices, kernel_size=2, stride=2, output_size=x2_size)
        x = self.block2b(x)

        # Stage 1d
        x = F.max_unpool2d(x, x1_maxpool_indices, kernel_size=2, stride=2, output_size=x1_size)
        x = self.block1b(x)

        del x1_size, x2_size, x3_size, x4_size, x5_size
        score = self.classifier(x)
        
        return score

    def init_vgg_weights(self):
        self.block1[0].weight = self.pretrained.features[0].weight
        self.block1[0].bias = self.pretrained.features[0].bias

        self.block1[3].weight = self.pretrained.features[2].weight
        self.block1[3].bias = self.pretrained.features[2].bias
        
        self.block2[0].weight = self.pretrained.features[5].weight
        self.block2[0].bias = self.pretrained.features[5].bias
        
        self.block2[3].weight = self.pretrained.features[7].weight
        self.block2[3].bias = self.pretrained.features[7].bias
           
        self.block3[0].weight = self.pretrained.features[10].weight
        self.block3[0].bias = self.pretrained.features[10].bias
        
        self.block3[3].weight = self.pretrained.features[12].weight
        self.block3[3].bias = self.pretrained.features[12].bias
        
        self.block3[6].weight = self.pretrained.features[14].weight
        self.block3[6].bias = self.pretrained.features[14].bias
        
        self.block4[0].weight = self.pretrained.features[17].weight
        self.block4[0].bias = self.pretrained.features[17].bias
        
        self.block4[3].weight = self.pretrained.features[19].weight
        self.block4[3].bias = self.pretrained.features[19].bias
        
        self.block4[6].weight = self.pretrained.features[21].weight
        self.block4[6].bias = self.pretrained.features[21].bias
        
        self.block5[0].weight = self.pretrained.features[24].weight
        self.block5[0].bias = self.pretrained.features[24].bias
        
        self.block5[3].weight = self.pretrained.features[26].weight
        self.block5[3].bias = self.pretrained.features[26].bias
        
        self.block5[6].weight = self.pretrained.features[28].weight
        self.block5[6].bias = self.pretrained.features[28].bias
        
        
if __name__ == "__main__":
    if test_model == True:
    
        batch_size, nclasses, h, w = 10, 20, 160, 160
        
        # test output size
        vgg_model = models.vgg16(pretrained=True)
        
        segnet_model = SEGNET(pretrained=vgg_model, nclasses=nclasses)
        input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
        output = segnet_model(input)
        assert output.size() == torch.Size([batch_size, nclasses, h, w])
        
        print("Passed size check")
        
        criterion = nn.BCELoss()
        optimizer = optim.SGD(segnet_model.parameters(), lr=1e-3, momentum=0.9)
        input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
        y = torch.autograd.Variable(torch.randn(batch_size, nclasses, h, w), requires_grad=False)
        for iter in range(10):
            optimizer.zero_grad()
            out = segnet_model(input)
            # print(out)
            out = torch.sigmoid(out)
            loss = criterion(out, y)
            loss.backward()
            # print(loss)
            print("iter{}, loss {}".format(iter, loss.item()))
            optimizer.step()