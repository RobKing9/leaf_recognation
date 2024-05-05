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
from hyperopt import hp, fmin, tpe, Trials

# 忽略警告
import warnings
warnings.filterwarnings("ignore")

# 主函数
def main():
    # 设置数据集路径
    input_path = './flavia'

    # 设置超参数搜索空间
    space = {
        'lr': hp.uniform('lr', 1e-3, 1e-2, 1e-4),
        'batch_size': hp.choice('batch_size', [32, 48, 64, 96]),
        'num_epochs': hp.choice('num_epochs', [50, 100])
    }

    # 加载图像、标签和特征
    images, labels, features = feature.load_images_and_labels(input_path)

    # 划分训练集、验证集和测试集
    train_val_X, test_X, train_val_y, test_y, train_val_features, test_features = (
        train_test_split(images, labels, features, test_size=0.2, random_state=42)
    )
    train_X, val_X, train_y, val_y, train_features, val_features = (
        train_test_split(train_val_X, train_val_y, train_val_features, test_size=0.25, random_state=42)
    )

    # 图像预处理操作
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 创建数据集对象
    train_dataset = ImageDataset(train_X, train_y, train_features, transform=transform)
    val_dataset = ImageDataset(val_X, val_y, val_features, transform=transform)

    # 定义超参数搜索函数
    def objective(params):
        batch_size = params['batch_size']
        num_epochs = params['num_epochs']
        lr = params['lr']

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型
        model = ResNet50(n_out=len(np.unique(labels)))
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        # 定义损失函数
        criterion = nn.CrossEntropyLoss()

        # 初始化最佳验证精度
        best_val_accuracy = 0.0

        # 记录每轮训练的损失和精度
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        # 开始训练
        for epoch in range(num_epochs):
            # 训练模式
            model.train()
            running_loss = 0.0
            corrects = 0
            total = 0
            for inputs, _, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = corrects.double() / total

            # 记录训练损失和精度
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc.item())

            # 验证模式
            running_loss = 0.0
            val_corrects = 0
            val_total = 0
            model.eval()
            with torch.no_grad():
                for inputs, _, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_corrects += torch.sum(preds == labels.data)
                    val_total += labels.size(0)
            val_epoch_loss = running_loss / val_total
            val_epoch_acc = val_corrects.double() / val_total

            # 记录验证损失和精度
            val_losses.append(val_epoch_loss)
            val_accs.append(val_epoch_acc.item())

            # 输出每轮训练的结果
            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
                  f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

            # 如果验证精度提升，则保存模型
            if val_epoch_acc > best_val_accuracy:
                best_val_accuracy = val_epoch_acc
                torch.save(model.state_dict(), './best_model.pth')
                print(f"Validation accuracy improved to {best_val_accuracy:.4f}, saving model...")

        # 返回最后一轮验证的精度作为优化目标
        return -val_accs[-1], train_losses, val_losses, train_accs, val_accs

    # 设置超参数优化器
    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=10, trials=trials)

    # 输出最佳参数
    print("Best hyperparameters:", best)

    # 绘制损失和精度曲线
    plt.figure(figsize=(12, 5))

    # 绘制不同学习率下的损失曲线
    plt.subplot(1, 2, 1)
    for trial in trials.trials:
        lr = trial['misc']['vals']['lr'][0]
        train_losses = trial['result']['train_losses']
        val_losses = trial['result']['val_losses']
        plt.plot(range(1, len(train_losses) + 1), train_losses, label=f'lr={lr}, Train')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label=f'lr={lr}, Val')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制不同学习率下的精度曲线
    plt.subplot(1, 2, 2)
    for trial in trials.trials:
        lr = trial['misc']['vals']['lr'][0]
        train_accs = trial['result']['train_accs']
        val_accs = trial['result']['val_accs']
        plt.plot(range(1, len(train_accs) + 1), train_accs, label=f'lr={lr}, Train')
        plt.plot(range(1, len(val_accs) + 1), val_accs, label=f'lr={lr}, Val')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
