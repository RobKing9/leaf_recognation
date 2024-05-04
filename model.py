import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import torchvision.models as models
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from skimage.feature import graycomatrix, graycoprops
# 定义卷积神经网络模型
class ResNet50(nn.Module):
    def __init__(self, n_out):
        super(ResNet50, self).__init__()
        # 使用 torchvision 库中预训练的 ResNet50 模型作为特征提取器
        self.features = models.resnet50(pretrained=True)
        # 将原始 ResNet50 模型的全连接层替换为一个标识层，即去掉原模型的分类层
        self.features.fc = nn.Identity()
        # 定义了一个包含多个层的序列模块，用于构建自定义的分类器
        # 这个分类器包括了两个全连接层、一个 ReLU 激活函数、一个 Dropout 层和一个 LogSoftmax 层，最终输出 n_out 个类别的概率分布
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_out),
            nn.LogSoftmax(dim=1)
        )
    # 定义前向传播过程，接受输入 x 并输出模型的预测结果
    def forward(self, x):
        # 前向传播过程
        x = self.features(x)
        x = self.classifier(x)
        return x
