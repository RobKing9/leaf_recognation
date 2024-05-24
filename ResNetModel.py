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
# from torchviz import make_dot
import torch
import torch.nn as nn
# 忽略警告
import warnings
warnings.filterwarnings("ignore")



# 定义一个残差块结构
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):  # 输入通道数，输出通道数，使能1x1卷积，步长
        super(Residual, self).__init__()

        # 定义瓶颈结构
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)  # 定义第一个卷积块
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  # 定义第二个卷积块

        # 定义1x1卷积块
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        # Batch归一化
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    # 定义前向传播路径
    def forward(self, x):
        y = nn.functional.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)

        return nn.functional.relu(y + x)

# 优化残差块结构
# 优化：使用瓶颈结构。使用1x1卷积减少通道数，然后使用3x3卷积进行特征提取，最后使用1x1卷积恢复通道数
# class Residual(nn.Module):
#     def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
#         super(Residual, self).__init__()
#
#         mid_channels = out_channels // 4  # 计算中间通道数
#
#         # 定义瓶颈结构
#         self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride)
#         self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
#
#         # 1x1卷积用于调整通道数和步幅
#         if use_1x1conv:
#             self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
#         else:
#             self.conv4 = None
#
#         self.bn1 = nn.BatchNorm2d(mid_channels)
#         self.bn2 = nn.BatchNorm2d(mid_channels)
#         self.bn3 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         y = nn.functional.relu(self.bn1(self.conv1(x)))
#         y = nn.functional.relu(self.bn2(self.conv2(y)))
#         y = self.bn3(self.conv3(y))
#         if self.conv4:
#             x = self.conv4(x)
#         return nn.functional.relu(y + x)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels  # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))

    return nn.Sequential(*blk)


# 定义一个全局平均池化层
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return nn.functional.avg_pool2d(x, kernel_size=x.size()[2:])  # 池化窗口形状等于输入图像的形状


# 定义ResNet18网络结构
def ResNet18():
    net = nn.Sequential(
        # 优化。将7x7卷积层改为5x5卷积层，减少参数数量和计算量
        nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=3),
        # nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", GlobalAvgPool2d())  # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
    net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, 32)))

    return net

# 定义ResNet34网络结构
def ResNet34():
    net = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    net.add_module("resnet_block1", resnet_block(64, 64, 3, first_block=True))  # 修改残差块数量
    net.add_module("resnet_block2", resnet_block(64, 128, 4))  # 修改残差块数量
    net.add_module("resnet_block3", resnet_block(128, 256, 6))  # 修改残差块数量
    net.add_module("resnet_block4", resnet_block(256, 512, 3))  # 修改残差块数量
    net.add_module("global_avg_pool", GlobalAvgPool2d())
    net.add_module("fc", nn.Sequential(nn.Flatten(), nn.Linear(512, 32)))
    return net
