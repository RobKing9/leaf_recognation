import pandas as pd
import torchvision.models
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time
from GoogLeNetData import train_val_split, train_data_process, val_data_process, test_data_process
from GoogLeNetModel import GoogLeNet
# 忽略警告
import warnings
warnings.filterwarnings("ignore")


img_dir = 'flavia/image/'
train_dir = 'flavia/train/'
val_dir = 'flavia/val/'
batch_size = 24

# 测试模型
def test_model(model, testdataloader, label, device):
    '''
    :param model: 网络模型
    :param testdataloader: 测试数据集
    :param label: 数据集标签
    :param device:
    '''

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

    # 计算并打印每个标签的测试识别准确率
    conf_mat = confusion_matrix(test_true, test_pre)
    label_accuracy = conf_mat.diagonal() / conf_mat.sum(axis=1)
    for i, accuracy in enumerate(label_accuracy):
        print(f"Accuracy for label {label[i]}: {accuracy * 100:.2f}%")

    # 计算混淆矩阵并可视化
    plt.figure(figsize=(10, 8))
    conf_mat = confusion_matrix(test_true, test_pre)
    df_cm = pd.DataFrame(conf_mat, index=label, columns=label)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.subplots_adjust(left=0.35, right=0.9, top=0.9, bottom=0.35)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    since = time.time()  # 当前时间
    train_val_split(img_dir, train_dir, val_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    googlenet = GoogLeNet()
    googlenet.load_state_dict(torch.load("GoogLeNet.pkl", map_location=device), strict=False)  # 加载最佳模型
    googlenet = googlenet.to(device)

    test_loader = test_data_process(val_dir)  # 加载测试集
    _, class_label = train_data_process(train_dir)  # 加载标签

    test_model(googlenet, test_loader, class_label, device)

    time_use = time.time() - since  # 计算耗费时间
    print("Tesl complete in {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))
