import torch
import torchvision.datasets  
from torch import nn  
from torch.nn import Linear  # 全连接层
from torch.utils.data import DataLoader 


dataset = torchvision.datasets.CIFAR10(
    root='./CIFAR10_dataset',  
    train=False,  
    download=True,  
    transform=torchvision.transforms.ToTensor() 
)


dataloader = DataLoader(dataset, batch_size=64)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义单个全连接层：输入维度196608，输出维度10
        self.linear1 = Linear(196608, 10)  # 196608 = 3 * 32 * 32 * 64（通道*高*宽*批量）

    def forward(self, input):
        output = self.linear1(input)  
        return output


net = Net()

for data in dataloader:
    imgs, targets = data  

    # 将当前批次的所有图像展平成1维向量
    # imgs形状原为[64, 3, 32, 32] -> 展平后变为[64 * 3 * 32 * 32] = [196608]
    # 也可以用reshape函数
    output = torch.flatten(imgs)

    output = net(output)  # 输出形状变为[10]


    print(output.shape)