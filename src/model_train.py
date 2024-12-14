import copy
import time

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import GoogLeNet, Inception
import torch.nn as nn
import pandas as pd


def train_val_data_process():
    # 定义数据集的路径
    ROOT_TRAIN = r'../data'

    # 定义数据集处理方法变量
    train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # 加载数据集
    train_data = ImageFolder(ROOT_TRAIN, transform=train_transform)

    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=32,
                                       shuffle=True,
                                       num_workers=2)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=32,
                                     shuffle=True,
                                     num_workers=2)

    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # 设定训练所用到的设备，有GPU用GPU没有GPU用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()
    # 将模型放入到训练设备中
    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)
        # 当前时间
        since = time.time()
        # 初始化参数
        # 训练集损失函数
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0
        # 验证集损失函数
        val_loss = 0.0
        # 验证集准确度
        val_corrects = 0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        # 对每一个mini-batch训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将特征放入到训练设备中
            b_x = b_x.to(device)
            # 将标签放入到训练设备中
            b_y = b_y.to(device)
            # 设置模型为训练模式
            model.train()

            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step()
            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于训练的样本数量
            train_num += b_x.size(0)
        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将特征放入到验证设备中
            b_x = b_x.to(device)
            # 将标签放入到验证设备中
            b_y = b_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)

            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度train_corrects加1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于验证的样本数量
            val_num += b_x.size(0)

        # 计算并保存每一次迭代的loss值和准确率
        # 计算并保存训练集的loss值
        train_loss_all.append(train_loss / train_num)
        # 计算并保存训练集的准确率
        train_acc_all.append(train_corrects.double().item() / train_num)

        # 计算并保存验证集的loss值
        val_loss_all.append(val_loss / val_num)
        # 计算并保存验证集的准确率
        val_acc_all.append(val_corrects.double().item() / val_num)

        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            # 保存当前最高准确度
            best_acc = val_acc_all[-1]
            # 保存当前最高准确度的模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练和验证的耗时
        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    # 选择最优参数，保存最优参数的模型
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, r"best_model.pth")

    return train_process


if __name__ == '__main__':
    # 加载需要的模型
    GoogLeNet = GoogLeNet(Inception)
    # 加载数据集
    train_data, val_data = train_val_data_process()
    # 利用现有的模型进行模型的训练
    train_process = train_model_process(GoogLeNet, train_data, val_data, num_epochs=150)
