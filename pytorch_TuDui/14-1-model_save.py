"""
两种保存/加载方式对比：
    方式一：保存整个模型对象
    优点：加载简单，不需要重新定义模型结构
    缺点：文件较大，可能受环境变化影响（如自定义类）
    方式二：仅保存状态字典（推荐）
    优点：文件小，跨环境兼容性好
    缺点：加载时需要先创建模型结构
自定义模型陷阱：
    保存自定义模型时，<<类定义(net)必须存在于加载环境中>>
    若加载时找不到类定义，会出现反序列化错误


    加载的时候记得带上weights_only参数，weights_only=True表示加载的是“仅权重文件”，weights_only=False表示加载的是“完整模型文件”，
"""

import torch
import torchvision
import torch.nn as nn

"""
vgg16_false=torchvision.models.vgg16()

# 保存方式一：保存整个模型（包含结构和参数）
torch.save(vgg16_false, "vgg16_method1.pth")


# 加载方式一：加载整个模型（不需要重新定义模型结构）
model_1 = torch.load("vgg16_method1.pth",weights_only=False)
print(model_1)  # 打印加载的模型结构
print("----------------------------------------------" \
"-------------------------------------------------------")

# 保存方式二：只保存模型参数（官方推荐，更轻量且兼容性更好）
torch.save(vgg16_false.state_dict(), "vgg16_method2.pth")


# 加载方式二：需要先创建相同结构的模型，再加载参数
# 1. 重新初始化一个相同结构的VGG16模型
vgg16_false1 = torchvision.models.vgg16()
# 2. 加载之前保存的参数到模型中
vgg16_false1.load_state_dict(torch.load("vgg16_method2.pth",weights_only=True))
print(vgg16_false)  # 打印加载后的模型结构

"""


# 自定义模型保存的陷阱演示
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义一个简单的卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        # 前向传播计算
        x = self.conv1(x)
        return x


# 创建自定义模型的实例
net = Net()  # 注意实例化名字还是要和类变量名字不同
# 保存自定义模型
torch.save(net, "net_test.pth")

# 加载自定义模型 - 注意陷阱：需要保证类定义在相同环境中可用
# model = torch.load("net1.pth")  # 若net类定义不可访问，加载会失败
