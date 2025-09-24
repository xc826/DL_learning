import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


input = torch.tensor([[1, -0.5], [-1, 3]], dtype=torch.float32)

# 重塑张量维度为神经网络标准输入格式
input = torch.reshape(input, (-1, 1, 2, 2))


dataset = torchvision.datasets.CIFAR10(
    root="./CIFAR10_dataset",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

dataLoader = DataLoader(dataset, batch_size=64)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.relu = ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output


net = Net()


writer = SummaryWriter("activation_function_log")
step = 0


for data in dataLoader:
    imgs, targets = data

    writer.add_images("input", imgs, global_step=step)

    output = net(imgs)

    writer.add_images("output", output, step)

    step += 1  # 增加步数计数器


writer.close()
