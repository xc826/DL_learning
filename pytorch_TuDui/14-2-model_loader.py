import torch
import torchvision
import torch.nn as nn

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


# # 创建自定义模型的实例
# net = net()
# # 保存自定义模型
# torch.save(net, "net1.pth")

# 加载自定义模型 - 注意陷阱：需要保证类定义在相同环境中可用
model = torch.load("net_test.pth",weights_only=False)  # 若net类定义不可访问，加载会失败
print(model)