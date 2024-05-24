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

img_dir = 'flavia/image/'
train_dir = 'flavia/train/'
val_dir = 'flavia/val/'
batch_size = 24
lr = 0.02
num_epochs = 100
num_workers = 8


train_data_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                            transforms.RandomVerticalFlip(p=0.5),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5],
                                                                 [0.5])
                                            ]
                                           )

val_data_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5],
                                                               [0.5])
                                          ]
                                         )

test_data_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5],
                                                               [0.5])
                                          ]
                                         )

# 分离训练集和验证集文件
def train_val_split(imgdir, traindir, valdir, split_rate=0.8):
    # 清空验证集文件
    classlist = os.listdir(valdir)
    for leaf_class in classlist:  # 读取所有的类别文件夹
        if os.listdir(valdir + leaf_class):  # 如果文件夹不为空，则清空文件夹
            imglist = os.listdir(valdir + leaf_class)
            for img_class in imglist:  # 读取每个类别文件夹中的图片
                os.remove(valdir + leaf_class + '/' + img_class)  # 删除图片

    # 清空训练集文件
    classlist = os.listdir(traindir)
    for leaf_class in classlist:  # 读取所有的类别文件夹
        if os.listdir(traindir + leaf_class):  # 如果文件夹不为空，则清空文件夹
            imglist = os.listdir(traindir + leaf_class)
            for img_class in imglist:  # 读取每个类别文件夹中的图片
                os.remove(traindir + leaf_class + '/' + img_class)  # 删除图片

    classlist = os.listdir(imgdir)
    for leaf_class in classlist:  # 读取所有的类别文件夹
        all_imgs = []
        imglist = os.listdir(imgdir + leaf_class)
        for img_class in imglist:  # 读取每个类别文件夹中的图片
            all_imgs.append(img_class)
        random.shuffle(all_imgs)  # 随机打乱图片顺序
        train_size = int(len(all_imgs) * split_rate)  # 按比例分割训练集和验证集
        val_size = len(all_imgs) - train_size
        assert (train_size > 0)
        assert (val_size > 0)

        train_imgs = all_imgs[:train_size]
        val_imgs = all_imgs[train_size:]

        for idx, imgs in enumerate(train_imgs):  # 移动图片到训练集文件夹中
            st.copy(imgdir + leaf_class + '/' + imgs, traindir + leaf_class)

        for idx, imgs in enumerate(val_imgs):  # 移动图片到验证集文件夹中
            st.copy(imgdir + leaf_class + '/' + imgs, valdir + leaf_class)

    print('dataset has been split.')


def train_data_process(train_data_path):
    train_data = ImageFolder(train_data_path, transform=train_data_transforms)
    train_data_loader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    class_label = train_data.classes  # 训练集的标签

    # 加载及可视化一个Batch的图像
    '''for step, (b_x, b_y) in enumerate(train_data_loader):
        if step > 0:
            break

    batch_x = b_x.squeeze().numpy()  # 将四维张量移除第1维，并转换成Numpy数组
    batch_y = b_y.numpy()  # 将张量转换成Numpy数组

    plt.figure(figsize=(12, 5))
    for ii in np.arange(len(batch_y)):
        plt.subplot(4, 16, ii+1)
        plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
        plt.title(class_label[batch_y[ii]], size=9)
        plt.axis("off")
        plt.subplots_adjust(wspace=0.05)
    plt.show()'''

    return train_data_loader, class_label


def val_data_process(val_data_path):
    val_data = ImageFolder(val_data_path, transform=val_data_transforms)
    val_data_loader = Data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return val_data_loader


# 处理测试集数据
def test_data_process(test_data_path):
    test_data = ImageFolder(test_data_path, transform=test_data_transforms)
    test_data_loader = Data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_data_loader
