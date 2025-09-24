import torchvision
import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


test_set = torchvision.datasets.CIFAR10(
    root="./CIFAR10_dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True
)


test_loader = DataLoader(dataset=test_set,batch_size=64)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        #创建卷积层
        self.conv1=Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x=self.conv1(x)
        return x


net=Net()

# 创建TensorBoard日志记录器
writer = SummaryWriter("convolution_log")  # 日志保存在"TensorBoard"目录
step = 0  # 初始化步数计数器

# 遍历数据加载器中的批次
for data in test_loader:
    # 解包数据：获取图像和标签
    imgs, targets = data

    # 将图像输入网络进行卷积操作
    output = net(imgs)

    # 记录输入图像到TensorBoard
    # 输入图像形状: [batch_size=64, channels=3, height=32, width=32]
    writer.add_images('input', imgs, step)

    # 处理输出图像以便可视化：
    # 卷积后输出形状: [64, 6, 30, 30] - 6通道无法直接显示
    # 重塑为3通道格式: [batch_size*2, 3, 30, 30]
    output = torch.reshape(output, (-1, 3, 30, 30))

    # 记录输出图像到TensorBoard
    writer.add_images('output', output, step)

    step += 1  # 增加步数计数器
    print(step)

# 关闭TensorBoard写入器
writer.close()