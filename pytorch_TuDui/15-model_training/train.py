import torchvision
import torch
from torch.utils.data import DataLoader
from model import *
from torch import nn
from torch.utils.tensorboard import SummaryWriter


# 准备数据集
train_data = torchvision.datasets.CIFAR10(
    root="../CIFAR10_dataset",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
test_data = torchvision.datasets.CIFAR10(
    root="../CIFAR10_dataset",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)


# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练集的长度是：{}".format(train_data_size))
print("测试集的长度是：{}".format(test_data_size))


# 利用DataLoader来加载数据集
train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)

# 创建网络模型
test = Test()

# 损失函数
loss_fn = nn.CrossEntropyLoss()


# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(test.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0

# 记录测试的次数
total_test_step = 0

# 训练的轮数
epoch = 10


# 添加tensorboard
writer = SummaryWriter("./logs_train")

for i in range(epoch):
    print("----------第{}轮训练开始----------".format(i + 1))

    # 训练步骤开始
    test.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = test(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    test.eval()
    total_test_loss = 0
    total_accuracy=0
    with torch.no_grad():  #测试时不需要调整梯度和优化
        for data in test_dataloader:
            imgs, targets = data
            outputs = test(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accuracy=(outputs.argmax(1)==targets).sum()
            total_accuracy+=accuracy
    print("整体测试集的Loss: {}".format(total_test_loss))
    print("整体测试集的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step += 1


    # 保存每一轮训练的模型
    # torch.save(test,"test_{}.pth".format(i+1)) # 方式1
    # 方式2，只保存模型参数，更轻量
    torch.save(test.state_dict(),"./model_save/test_{}.pth".format(i)) 
    print("模型已保存")


writer.close()
