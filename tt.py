import numpy as np
import feature
from model import ResNet50
from data import ImageDataset
import torch
import torch.nn as nn
import torch.optim as optim
from multiprocessing import Pool
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
# 忽略警告
import warnings
warnings.filterwarnings("ignore")

def load_data(input_path):
    images, labels, features = feature.load_images_and_labels(input_path)
    return images, labels, features

def preprocess_data(data):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images, labels, features = data
    images = [transform(img) for img in images]
    return images, labels, features

def evaluate_model(input_path, model_path='./best_model.pth', batch_size=32):
    # 加载数据
    with Pool(processes=3) as pool:
        images, labels, features = pool.map(load_data, [input_path])[0]
    # 划分数据
    train_val_X, test_X, train_val_y, test_y, train_val_features, test_features = train_test_split(
        images, labels, features, test_size=0.2, random_state=42
    )
    # 预处理数据
    with Pool(processes=3) as pool:
        test_X, _, test_features = pool.map(preprocess_data, [(test_X, test_y, test_features)])[0]

    # 创建DataLoader
    test_dataset = ImageDataset(test_X, test_y, test_features)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    n_out = len(np.unique(labels))
    model = ResNet50(n_out=n_out)
    # 加载最佳模型
    print("loaded model from best_model.pth...")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 评估模型
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, _, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    test_accuracy = test_correct / test_total
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # 输出测试结果
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')

if __name__ == '__main__':
    input_path = './flavia'
    evaluate_model(input_path)
