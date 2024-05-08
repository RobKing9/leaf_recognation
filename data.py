# import os
# import cv2
import numpy as np
import torch
import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import torchvision.models as models
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from skimage.feature import graycomatrix, graycoprops



# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, images, labels, features, transform=None):
        self.images = images
        self.labels = labels
        self.features = features
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # 提取图像特征
        features = self.features[idx]
        # 如果定义了转换函数，则对图像进行转换
        if self.transform:
            image = self.transform(image)
        return image, features, label

