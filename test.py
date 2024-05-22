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
                                            transforms.RandomRotation(degrees=180),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.769235, 0.869587, 0.733468],
                                                                 [0.339912, 0.204988, 0.388254])
                                            ]
                                           )

val_data_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                          transforms.RandomVerticalFlip(p=0.5),
                                          transforms.RandomHorizontalFlip(p=0.5),
                                          transforms.RandomRotation(degrees=180),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.769235, 0.869587, 0.733468],
                                                               [0.339912, 0.204988, 0.388254])
                                          ]
                                         )

test_data_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                           transforms.RandomVerticalFlip(p=0.5),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.RandomRotation(degrees=180),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.769235, 0.869587, 0.733468],
                                                                [0.339912, 0.204988, 0.388254])
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
# def test_data_process(test_data_path):
#     test_data = ImageFolder(test_data_path, transform=test_data_transforms)
#     test_data_loader = Data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#
#     return test_data_loader


def test_data_process(test_data_path):
    test_data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"The directory {test_data_path} does not exist.")

    for class_name in os.listdir(test_data_path):
        class_path = os.path.join(test_data_path, class_name)
        if os.path.isdir(class_path):
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(
                ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'))]
            if not image_files:
                raise FileNotFoundError(f"No valid image files found in the directory {class_path}.")
            else:
                print(f"Found {len(image_files)} images in {class_path}.")

    test_data = ImageFolder(test_data_path, transform=test_data_transforms)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    return test_loader


# 优化残差块结构
# 优化：使用瓶颈结构。使用1x1卷积减少通道数，然后使用3x3卷积进行特征提取，最后使用1x1卷积恢复通道数
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()

        mid_channels = out_channels // 4  # 计算中间通道数

        # 定义瓶颈结构
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

        # 1x1卷积用于调整通道数和步幅
        if use_1x1conv:
            self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv4 = None

        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = nn.functional.relu(self.bn1(self.conv1(x)))
        y = nn.functional.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        if self.conv4:
            x = self.conv4(x)
        return nn.functional.relu(y + x)


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

# def test_model(model, testdataloader, label, device):
#     test_corrects = 0.0
#     test_num = 0
#     test_acc = 0.0
#     test_true = []
#     test_pre = []
#
#     with torch.no_grad():
#         for test_data_x, test_data_y in testdataloader:
#             test_data_x = test_data_x.to(device)
#             test_data_y = test_data_y.to(device)
#             model.eval()  # 设置模型为评估模式，不启用Batch Normalization和Dropout
#             output = model(test_data_x)  # 前向传播过程，输入为测试数据集，输出为对每个样本的预测
#             pre_lab = torch.argmax(output, 1)  # 查找每一行中最大值对应的行标
#             test_corrects += torch.sum(torch.eq(pre_lab, test_data_y.data))
#             test_num += test_data_x.size(0)  # 当前用于训练的样本数量
#             test_true += test_data_y.cpu().tolist()
#             test_pre += pre_lab.cpu().tolist()
#
#     test_acc = test_corrects.item() / test_num
#     precision = precision_score(test_true, test_pre, average='macro')
#     recall = recall_score(test_true, test_pre, average='macro')
#     f1 = f1_score(test_true, test_pre, average='macro')
#     # 计算 准确率, 精确率，召回率，F1值 四个指标
#     print("test accuracy:", test_acc)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     print("F1 score:", f1)
#
#     # 计算混淆矩阵并可视化
#     plt.figure(figsize=(10, 8))
#     conf_mat = confusion_matrix(test_true, test_pre)
#     df_cm = pd.DataFrame(conf_mat, index=label, columns=label)
#     heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
#     heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
#     heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.subplots_adjust(left=0.35, right=0.9, top=0.9, bottom=0.35)
#     plt.tight_layout()
#     plt.show()


def test_model(model, testdataloader, label, device):
    test_corrects = 0.0
    test_num = 0
    test_acc = 0.0
    test_true = []
    test_pre = []

    with torch.no_grad():
        for test_data_x, test_data_y in testdataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()  # 设置模型为评估模式，不启用Batch Normalization和Dropout
            output = model(test_data_x)  # 前向传播过程，输入为测试数据集，输出为对每个样本的预测
            pre_lab = torch.argmax(output, 1)  # 查找每一行中最大值对应的行标
            test_corrects += torch.sum(torch.eq(pre_lab, test_data_y.data))
            test_num += test_data_x.size(0)  # 当前用于训练的样本数量
            test_true += test_data_y.cpu().tolist()
            test_pre += pre_lab.cpu().tolist()

    test_acc = test_corrects.item() / test_num
    precision = precision_score(test_true, test_pre, average='macro')
    recall = recall_score(test_true, test_pre, average='macro')
    f1 = f1_score(test_true, test_pre, average='macro')
    # 计算 准确率, 精确率，召回率，F1值 四个指标
    print("test accuracy:", test_acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1)

    # 计算混淆矩阵
    conf_mat = confusion_matrix(test_true, test_pre)

    # 计算每个标签的准确率
    label_accuracy = conf_mat.diagonal() / conf_mat.sum(axis=1)

    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    df_cm = pd.DataFrame(conf_mat, index=label, columns=label)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')

    # 在图中显示每个标签的准确率
    for i, acc in enumerate(label_accuracy):
        heatmap.text(i, i, f'{acc:.2f}', color='red', ha='center', va='center')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.subplots_adjust(left=0.35, right=0.9, top=0.9, bottom=0.35)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    train_val_split(img_dir, train_dir, val_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    resnet = ResNet18()
    resnet.load_state_dict(torch.load("Resnet18_improve.pkl", map_location=device))  # 加载最佳模型
    resnet = resnet.to(device)

    test_loader = test_data_process(val_dir)  # 加载测试集
    _, class_label = train_data_process(train_dir)  # 加载标签

    test_model(resnet, test_loader, class_label, device)
