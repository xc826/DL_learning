import torch
from torch import nn
from torch.nn import *
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )
   
    def forward(self, x):
        x = self.module(x)
        return x


net = Net()

input = torch.ones((64, 3, 32, 32))

output = net(input)

print(output.shape)  # 输出：torch.Size([64, 10])


writer = SummaryWriter("network_build_log")

writer.add_graph(net, input)  # 记录网络结构和输入数据

writer.close()
