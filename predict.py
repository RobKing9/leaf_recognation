import torch
from torchvision import transforms
from PIL import Image
from model import ResNet50

def load_model(model_path):
    model = ResNet50(n_out=32)  # 替换为你的分类类别数
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image(image_path, model, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 在第0维添加一个维度，因为模型输入要求是batch x channels x height x width
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()  # 返回预测的类别索引

if __name__ == "__main__":
    # 指定最佳模型的路径
    model_path = './best_model.pth'
    # 加载模型
    model = load_model(model_path)
    # 定义图像预处理操作
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 指定输入图像的路径
    image_path = './test_image.jpg'
    # 预测图像标签
    predicted_label = predict_image(image_path, model, transform)
    print("Predicted label:", predicted_label)
