import torch
from torchvision import transforms

from model import GoogLeNet, Inception
from PIL import Image
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    # 加载模型
    model = GoogLeNet(Inception)
    model.load_state_dict(torch.load('best_model.pth'))
    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    classes = [1, 0, 0, 0]
    p = Path(r'../Data')
    s = []

    for file in p.glob('*'):  # 正则表达式
        if file.suffix in ['.jpg', '.jpeg']:
            image = Image.open(file)
            image = image.convert('RGB')
            a = file.stem
            # 定义数据集处理方法变量
            test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            image = test_transform(image)

            # 添加批次维度
            image = image.unsqueeze(0)
            with torch.no_grad():
                model.eval()
                image = image.to(device)
                output = model(image)
                pre_lab = torch.argmax(output, dim=1)
                result = pre_lab.item()
                b = classes[result]
            s.append([a, b])

    result = pd.DataFrame(s)
    result.columns = ['id', 'label']
    result = result.sort_values(by='id')
    result.to_csv('../cla_pre.csv', index=False, header=False)
