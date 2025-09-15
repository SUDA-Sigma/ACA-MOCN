#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
import numpy as np
import time

class Focal_L2(nn.Module):
    def __init__(self, gamma):
        super(Focal_L2, self).__init__()
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma

    def forward(self, input, target):

        lossB = [] # Batch Loss
        for pre, gt in zip(input, target):
            lossC = [] # channel Loss

            weightC = [] # wei
            for pre_c, gt_c in zip(pre, gt):
                preCount = pre_c.sum()
                gtCount = gt_c.sum()
                loss = self.loss_fn(pre_c, gt_c)
                weight = torch.exp(torch.log1p(torch.abs(preCount - gtCount) / torch.abs(gtCount + 1)) * self.gamma)
                lossC.append(loss*weight)

            lossB.append(sum(lossC))
        return sum(lossB)

if __name__ == '__main__':
    MSE = nn.MSELoss()

    data = torch.randn(2, 6, 64, 64)
    # data += 2
    target = torch.randn(2, 6, 64, 64)

    data = Variable(data, requires_grad=True)
    target = Variable(target)

    model = Focal_L2(0.1)
    t1 = time.time()
    lossFocal = model(data, target)
    lossMSE = MSE(data, target)
    t2 = time.time()
    print(lossFocal)
    lossFocal.backward()
    t3 = time.time()
    print(t2 - t1)
    print(t3 - t2)
    print(t3 - t1)
    print("lossFocal: " + str(lossFocal))
    print("lossMSE: " + str(lossMSE))
    print(data.grad)
