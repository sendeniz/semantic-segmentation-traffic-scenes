#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:39:09 2022

@author: deniz
"""

import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG

class FCNN8(nn.Module):
    def __init__(self, pretrained, nclasses):
        super().__init__()
        self.nclasses = nclasses
        self.pretrained = pretrained
        
        
        self.block1 = nn.Sequential( 
           nn.ConvTranspose2d(in_channels = 512,out_channels = 512, 
                     kernel_size = (3, 3), stride = 2,
                     padding = 1, dilation = 1,  output_padding=1),
           nn.ReLU(inplace=True)
           )
           
        self.block2 = nn.Sequential(
             nn.BatchNorm2d(512),
             nn.ConvTranspose2d(512, 256, kernel_size = (3, 3), stride = 2,
                                padding = 1, dilation = 1,  output_padding=1),
             nn.ReLU(inplace=True),
             )
           
        self.block3 = nn.Sequential(
             nn.BatchNorm2d(256),
             nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=2, 
                                padding=1, dilation=1, output_padding=1),
             nn.ReLU(inplace=True),
             nn.BatchNorm2d(128),
             )
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2, 
                               padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            )
        
        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=2, 
                               padding=1, dilation=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            )
        
        self.classifier = nn.Sequential(
            nn.Conv2d(32, nclasses, kernel_size=(1, 1))
            )
    def forward(self,x):
        out = self.pretrained(x)
        x3 = out['x3']
        x4 = out['x4']
        x5 = out['x5']
        
        score = self.block1(x5)
        score = self.block2(score + x4)
        score = self.block3(score + x3)
        score = self.block4(score)
        score = self.block5(score)
        score = self.classifier(score)
        
        return score 
        
class VGGNet(VGG):
    def __init__(self, pretrained = True, model='vgg16', requires_grad=True, remove_fc=True):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]
        
        if pretrained == True:
           exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)
                         
        if remove_fc:
            del self.classifier


    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output

    
ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == "__main__":
    batch_size, nclasses, h, w = 10, 20, 160, 160
    
    # test output size
    vgg_model = VGGNet(requires_grad=True)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, 224, 224))
    output = vgg_model(input)
    assert output['x5'].size() == torch.Size([batch_size, 512, 7, 7])
    
    fcn_model = FCNN8(pretrained=vgg_model, nclasses=nclasses)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    output = fcn_model(input)
    assert output.size() == torch.Size([batch_size, nclasses, h, w])
    
    print("Passed size check")
    
    fcn_model = FCNN8(pretrained=vgg_model, nclasses=nclasses)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-3, momentum=0.9)
    input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    y = torch.autograd.Variable(torch.randn(batch_size, nclasses, h, w), requires_grad=False)
    for iter in range(10):
        optimizer.zero_grad()
        out = fcn_model(input)
        # print(out)
        out = torch.sigmoid(out)
        loss = criterion(out, y)
        loss.backward()
        # print(loss)
        print("iter{}, loss {}".format(iter, loss.item()))
        optimizer.step()