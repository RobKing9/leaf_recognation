import numpy as np
import pandas as pd
import torchvision.models
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import shutil as st
import copy
import time
from torchviz import make_dot
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from torchvision import transforms
from torchvision import models
from torchvision.datasets import ImageFolder
# 忽略警告
import warnings
warnings.filterwarnings("ignore")

# 定义一个全局平均池化层
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return nn.functional.avg_pool2d(x, kernel_size=x.size()[2:])  # 池化窗口形状等于输入图像的形状


# 定义一个Inception模块
class Inception(nn.Module):
    def __init__(self, in_ch, ch1, ch2, ch3, ch4):  # 每一条线路中的输入输出通道数
        super(Inception, self).__init__()
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_ch, ch1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_ch, ch2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(ch2[0], ch2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_ch, ch3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(ch3[0], ch3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大池化层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_ch, ch4, kernel_size=1)

    def forward(self, x):
        p1 = nn.functional.relu(self.p1_1(x))
        p2 = nn.functional.relu(self.p2_2(nn.functional.relu(self.p2_1(x))))
        p3 = nn.functional.relu(self.p3_2(nn.functional.relu(self.p3_1(x))))
        p4 = nn.functional.relu(self.p4_2(self.p4_1(x)))

        return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维度上进行拼接


# 定义GoogLeNet模型网络结构
def GoogLeNet():
    b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                       )

    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                       nn.Conv2d(64, 192, kernel_size=3, padding=1),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                       )

    b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                       Inception(256, 128, (128, 192), (32, 96), 64),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                       )

    b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                       Inception(512, 160, (112, 224), (24, 64), 64),
                       Inception(512, 128, (128, 256), (24, 64), 64),
                       Inception(512, 112, (144, 288), (32, 64), 64),
                       Inception(528, 256, (160, 320), (32, 128), 128),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                       )

    b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                       Inception(832, 384, (192, 384), (48, 128), 128),
                       GlobalAvgPool2d()
                       )

    net = nn.Sequential(b1,
                        b2,
                        b3,
                        b4,
                        b5,
                        nn.Flatten(),
                        nn.Linear(1024, 32)
                        )

    return net
