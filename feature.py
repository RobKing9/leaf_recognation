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

