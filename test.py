import os
import cv2
import numpy as np
import feature
from model import ResNet50
from data import ImageDataset
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
# 忽略警告
import warnings
warnings.filterwarnings("ignore")

def main():
    # 设置数据集路径
    input_path = './flavia'
    # 参数
    batch_size = 32
    # 加载图像、标签和特征
    images, labels, features = feature.load_images_and_labels(input_path)
    # 划分训练集、验证集和测试集
    train_val_X, test_X, train_val_y, test_y, train_val_features, test_features = (
        train_test_split(images, labels, features, test_size=0.2, random_state=42)
    )

    # 图像预处理操作
    transform = transforms.Compose([
        transforms.ToPILImage(),  # 将输入的数据转换为PIL Image格式
        transforms.Resize(256),  # 调整图像大小为256x256
        transforms.CenterCrop(224),  # 中心剪裁，将图像裁剪为224x224像素大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 图像归一化
    ])

    # 创建数据集对象
    test_dataset = ImageDataset(test_X, test_y, test_features, transform=transform)

    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 获取类别数量
    n_out = len(np.unique(labels))
    # 初始化模型
    model = ResNet50(n_out=n_out)
    # 加载最佳模型
    model.load_state_dict(torch.load('./best_model.pth'))
    model.eval()     # 设置为评估模式

    # 在测试集上进行测试
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():    # 关闭梯度计算
        for inputs, _, labels in test_loader:
            outputs = model(inputs)    # 预测输出
            _, predicted = torch.max(outputs, 1)    # 预测类别
            test_total += labels.size(0)     # 累计样本数量
            test_correct += (predicted == labels).sum().item()  # 累计正确预测数量
            all_predictions.extend(predicted.cpu().numpy())  # 保存预测结果
            all_labels.extend(labels.cpu().numpy())  # 保存标签结果

    # 计算测试集上的指标
    test_accuracy = test_correct / test_total    # 准确率
    precision = precision_score(all_labels, all_predictions, average='macro')    # 精确率
    recall = recall_score(all_labels, all_predictions, average='macro')    # 召回率
    f1 = f1_score(all_labels, all_predictions, average='macro')    # F1值
    conf_matrix = confusion_matrix(all_labels, all_predictions)      # 混淆矩阵

    # 输出测试结果
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')


if __name__ == '__main__':
    main()