from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from model import ResNet50
import torchvision.transforms as transforms
from torch.autograd import Variable as V
import torch as t
# 忽略警告
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# totensor 转换
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def prediect(img):
    print('wait..')
    classes = ['Anhui Barberry', 'Beale\'s barberry', 'Big-fruited Holly', 'Canadian poplar', 'Chinese Toon', 'Chinese cinnamon',
              'Chinese horse chestnut', 'Chinese redbud', 'Chinese tulip tree', 'Crape myrtle, Crepe myrtle', 'Ford Woodlotus',
              'Glossy Privet', 'Japan Arrowwood', 'Japanese Flowering Cherry', 'Japanese cheesewood', 'Japanese maple', 'Nanmu',
              'camphortree', 'castor aralia', 'deodar', 'ginkgo, maidenhair tree', 'goldenrain tree', 'oleander', 'peach',
              'pubescent bamboo', 'southern magnolia', 'sweet osmanthus', 'tangerine', 'trident maple', 'true indigo',
              'wintersweet', 'yew plum pine']

    #读入图片
    #img = Image.open('图片路径')
    img=trans(img)        #这里经过转换后输出的input格式是[C,H,W],网络输入还需要增加一维批量大小B
    img = img.unsqueeze(0)      #增加一维，输出的img格式为[1,C,H,W]

    model = ResNet50(32).to(device)   #导入网络模型
    model.eval()
    model.load_state_dict(t.load('./best_model.pth'))        #加载训练好的模型文件

    input = V(img.to(device))
    score = model(input)            #将图片输入网络得到输出
    probability = t.nn.functional.softmax(score,dim=1)      #计算softmax，即该图片属于各类的概率
    max_value,index = t.max(probability,1)          #找到最大概率对应的索引号，该图片即为该索引号对应的类别
    #print(index)
    msg = '{} 可能是：{}'.format('这张图',classes[index])
    print(msg)
    return msg

# ------------------------------------------------------------------------

if __name__ == '__main__':

    img = Image.open('../test/1.jpg')
    plt.imshow(img)
    plt.show()
    prediect(img)
