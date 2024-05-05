import os
import cv2
import feature
from model import ResNet50
from data import ImageDataset
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
import matplotlib.pyplot as plt
# 忽略警告
import warnings
warnings.filterwarnings("ignore")

# 主函数
def main():
    # 设置数据集路径
    input_path = './flavia'
    # 参数
    batch_size = 48
    lr = 0.00001
    # 设置训练轮数
    num_epochs = 50
    # 初始化记录性能指标的列表
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # 加载图像、标签和特征
    images, labels, features = feature.load_images_and_labels(input_path)

    # 划分训练集、验证集和测试集
    # 20%作为测试集，其余作为训练集和验证集
    # train_val_X是训练验证集的图像数据，test_X是测试集的图像数据
    # train_val_y是训练验证集的标签数据,test_y是测试集的标签数据
    # train_val_features是训练验证集的特征数据，test_features是测试集的特征数据
    train_val_X, test_X, train_val_y, test_y, train_val_features, test_features = (
        train_test_split(images, labels, features, test_size=0.2, random_state=42)
    )
    # 再次划分训练集和验证集，75%作为训练集，25%作为验证集
    # train_X是训练集的图像数据，val_X是验证集的图像数据
    # train_y是训练集的标签数据，val_y是验证集的标签数据
    # train_features是训练集的特征数据，val_features是验证集的特征数据
    train_X, val_X, train_y, val_y, train_features, val_features = (
        train_test_split(train_val_X, train_val_y, train_val_features, test_size=0.25, random_state=42)
    )

    # 图像预处理操作
    transform = transforms.Compose([
        transforms.ToPILImage(),    # 将输入的数据转换为PIL Image格式
        transforms.Resize(256),     # 调整图像大小为256x256
        transforms.CenterCrop(224),     # 中心剪裁，将图像裁剪为224x224像素大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    # 图像归一化, 加速模型收敛
    ])

    # 创建数据集对象
    train_dataset = ImageDataset(train_X, train_y, train_features, transform=transform)
    val_dataset = ImageDataset(val_X, val_y, val_features, transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 获取类别数量
    n_out = len(np.unique(labels))
    # 初始化模型
    model = ResNet50(n_out=n_out)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 交叉熵损失函数适用于多分类问题，且每个类别的概率分布可以直接用softmax函数计算
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)


    # 初始化最佳验证精度
    best_val_accuracy = 0.0
    # 开始训练
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        for inputs, _, labels in train_loader:
            optimizer.zero_grad()    # 梯度清零
            outputs = model(inputs)    # 前向传播
            loss = criterion(outputs, labels)    # 计算损失
            loss.backward()    # 反向传播
            optimizer.step()    # 更新参数
            running_loss += loss.item()  # 累计损失
            _, preds = torch.max(outputs, 1)    # 对每个输入样本的预测类别
            corrects += torch.sum(preds == labels.data)    # 统计正确的预测数
            total += labels.size(0)
        # 计算每个epoch的平均损失和准确率
        epoch_loss = running_loss / len(train_loader)    # 计算每个epoch的平均损失
        epoch_acc = corrects.double() / total    # 计算每个epoch的平均准确率
        train_losses.append(epoch_loss)      # 记录训练损失
        train_accs.append(epoch_acc.item())      # 记录训练准确率

        # 验证模式
        running_loss = 0.0
        val_corrects = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            for inputs, _, labels in val_loader:
                outputs = model(inputs)    # 前向传播
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)    # 累计损失
                _, preds = torch.max(outputs, 1)        # 对每个输入样本的预测类别
                val_corrects += torch.sum(preds == labels.data)    # 统计正确的预测数
                val_total += labels.size(0)    # 统计样本总数

        # 计算每个epoch的平均验证损失和准确率
        val_epoch_loss = running_loss / val_total
        val_epoch_acc = val_corrects.double() / val_total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc.item())

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}, '
            f'Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_acc:.4f}')
        # 如果验证精度提升，则保存模型
        if val_epoch_acc > best_val_accuracy:
            best_val_accuracy = val_epoch_acc
            torch.save(model.state_dict(), './best_model.pth')
            print(f"Validation accuracy improved to {best_val_accuracy:.4f}, saving model...")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)    # 绘制两张子图
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()    # 显示图例

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()  # 调整子图间距
    plt.show()

# 执行主函数
if __name__ == "__main__":
    main()
