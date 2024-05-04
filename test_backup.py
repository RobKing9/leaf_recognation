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

# 忽略警告
import warnings

warnings.filterwarnings("ignore")


# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self, n_out):
        super(CNN, self).__init__()
        # 使用预训练的ResNet50模型的特征提取部分
        self.features = models.resnet50(pretrained=True)
        # 将原始模型的全连接层替换为一个标识层
        self.features.fc = nn.Identity()
        # 添加自定义分类器
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_out),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # 前向传播过程
        x = self.features(x)
        x = self.classifier(x)
        return x


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


# 加载图像并提取特征
def load_images_and_labels(input_path, img_size=(256, 256)):
    images = []
    labels = []
    features = []
    label_encoder = LabelEncoder()
    # 遍历数据集目录
    for dirname, _, filenames in os.walk(input_path):
        for filename in filenames:
            img_path = os.path.join(dirname, filename)
            # print(img_path)
            # 读取图像并转换为RGB格式
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 将图像调整为指定大小
            img = cv2.resize(img, img_size)
            images.append(img)
            # 提取图像特征
            lbp_feature = extract_lbp(img)
            gabor_feature = extract_gabor(img)
            # glcm_feature = extract_glcm(img)
            # fourier_feature = extract_fourier(img)
            combined_feature = np.concatenate((lbp_feature, gabor_feature))
            features.append(combined_feature)
            # 提取标签
            label = os.path.basename(dirname)
            labels.append(label)

    # 对标签进行编码
    labels = label_encoder.fit_transform(labels)
    return np.array(images), np.array(labels), np.array(features)


# 提取图像的LBP特征
def extract_lbp(image):
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 提取LBP特征
    lbp = local_binary_pattern(gray_image, 8, 1, method='uniform')
    # 计算直方图
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
    # 归一化直方图
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


# 提取图像的Gabor特征
def extract_gabor(image):
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gabor_feats = []
    # 设置Gabor滤波器参数
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    frequencies = [0.1, 0.3, 0.5]
    # 遍历所有方向和频率的滤波器
    for theta in thetas:
        for frequency in frequencies:
            # 获取Gabor滤波器
            kernel = cv2.getGaborKernel((3, 3), 1.0, theta, frequency, 0.5, 0, ktype=cv2.CV_32F)
            # 对图像进行滤波
            filtered = cv2.filter2D(gray_image, cv2.CV_32F, kernel)
            # 计算滤波后的均值作为特征
            gabor_feats.append(np.mean(filtered))
    return gabor_feats


# 提取图像的灰度共生矩阵特征
# def extract_glcm(image):
#     # 将图像转换为灰度图像
#     gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     # 计算灰度共生矩阵
#     glcm = graycomatrix(gray_image, [1], [0], symmetric=True, normed=True)
#     # 提取统计特征
#     contrast = graycoprops(glcm, 'contrast')
#     dissimilarity = graycoprops(glcm, 'dissimilarity')
#     homogeneity = graycoprops(glcm, 'homogeneity')
#     energy = graycoprops(glcm, 'energy')
#     correlation = graycoprops(glcm, 'correlation')
#     asm = graycoprops(glcm, 'ASM')
#     # 返回特征向量
#     return np.array(
#         [contrast[0, 0], dissimilarity[0, 0], homogeneity[0, 0], energy[0, 0], correlation[0, 0], asm[0, 0]])


# 提取图像的傅里叶描述子特征
# def extract_fourier(image):
#     # 将图像转换为灰度图像
#     gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     # 进行二维傅里叶变换
#     f_transform = np.fft.fft2(gray_image)
#     # 对频谱进行中心化
#     f_transform_shift = np.fft.fftshift(f_transform)
#     # 计算幅度谱并展平为一维向量
#     magnitude_spectrum = np.abs(f_transform_shift)
#     return magnitude_spectrum.ravel()


# 主函数
def main():
    # 设置数据集路径
    input_path = '../test/flavia'
    # 加载图像、标签和特征
    images, labels, features = load_images_and_labels(input_path)

    # 划分训练集、验证集和测试集
    train_val_X, test_X, train_val_y, test_y, train_val_features, test_features = train_test_split(images, labels,
                                                                                                   features,
                                                                                                   test_size=0.2,
                                                                                                   random_state=42)
    train_X, val_X, train_y, val_y, train_features, val_features = train_test_split(train_val_X, train_val_y,
                                                                                    train_val_features, test_size=0.25,
                                                                                    random_state=42)

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
    test_dataset = ImageDataset(test_X, test_y, test_features, transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 获取类别数量
    n_out = len(np.unique(labels))
    # 初始化模型
    cnn = CNN(n_out=n_out)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.0001, weight_decay=1e-4)

    # 设置训练轮数
    num_epochs = 20

    # 初始化最佳验证精度
    best_val_accuracy = 0.0
    # 开始训练
    for epoch in range(num_epochs):
        # 训练模式
        cnn.train()
        running_loss = 0.0
        for inputs, _, labels in train_loader:
            optimizer.zero_grad()
            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 验证模式
        val_correct = 0
        val_total = 0
        cnn.eval()
        with torch.no_grad():
            for inputs, _, labels in val_loader:
                outputs = cnn(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # 计算验证集精度
        val_accuracy = val_correct / val_total
        print(f"Epoch {epoch}: Validation Accuracy: {val_accuracy:.4f}")

        # 如果验证精度提升，则保存模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(cnn.state_dict(), '../test/best_model.pth')
            print(f"Validation accuracy improved to {best_val_accuracy:.4f}, saving model...")

    # 加载最佳模型
    cnn.load_state_dict(torch.load('../test/best_model.pth'))
    cnn.eval()

    # 在测试集上进行测试
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, _, labels in test_loader:
            outputs = cnn(inputs)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算测试集上的指标
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


# 执行主函数
if __name__ == "__main__":
    main()
