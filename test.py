#!/usr/bin/env python3    
# -*- coding: utf-8 -*- 
"""
    Time    : 2021/6/1 14:58
    Author  : 春晓
    Software: PyCharm
"""
import numpy
import numpy as np
import torch

n = 630
indices = torch.randint(0,n,size=(15,15))

label = 155
m = 10
labels = torch.zeros((m,))
labels += label
print(labels)

y = torch.tensor([1,2,3,4])
y_hat = torch.ones(4,10)
loss = torch.nn.CrossEntropyLoss()

print(y.shape)
print(y_hat.shape)
l = loss(y_hat,y)
print(l)

y1 = torch.zeros((60,))
print(y1.shape)
y1_hat = torch.ones((60,657))
print(y1_hat.shape)
l2 = loss(y1_hat,y1)
print(l2.shape)
